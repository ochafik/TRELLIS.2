"""
Pure OpenGL Mesh Renderer using moderngl.

This renderer bypasses nvdiffrast's HIP rasterizer which has multi-bin
race conditions on AMD GPUs. It uses moderngl for pure OpenGL rendering,
allowing full resolution (1024px+) on AMD.

Trade-offs:
- Slower than HIP due to CPU-GPU data transfers
- But works at any resolution without crashes
- Suitable for final export, not training

TRELLIS.2 AMD Port: Updated to use TRELLIS.2 mesh types.
"""

import os
import torch
import numpy as np
from easydict import EasyDict as edict
from ..representations.mesh import Mesh, MeshWithVoxel, MeshWithPbrMaterial

# Lazy import to avoid loading moderngl if not needed
_mgl_ctx = None


# Shader sources
VERTEX_SHADER = """
#version 330
uniform mat4 u_mvp;
in vec3 i_position;
in vec3 i_attr;
out vec3 v_attr;
void main() {
    gl_Position = u_mvp * vec4(i_position, 1.0);
    v_attr = i_attr;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 v_attr;
out vec3 f_color;
void main() {
    f_color = v_attr;
}
"""


class ModernGLContext:
    """Wrapper around moderngl context with shader programs."""

    def __init__(self):
        import moderngl
        self.mgl = moderngl
        self.ctx = moderngl.create_context(standalone=True)
        self._prog = None

    def get_program(self):
        if self._prog is None:
            self._prog = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=FRAGMENT_SHADER
            )
        return self._prog

    def rasterize(self, vertices, faces, attr, width, height, mvp, cull_backface=False):
        """
        Rasterize triangles with attributes.

        Args:
            vertices: [N, 3] float32 vertices
            faces: [F, 3] int32 face indices
            attr: [N, 3] float32 per-vertex attributes
            width, height: output resolution
            mvp: [4, 4] float32 MVP matrix
            cull_backface: whether to cull back faces

        Returns:
            image: [H, W, 3] float32 rendered image
            depth: [H, W] float32 depth buffer (0-1 range)
        """
        prog = self.get_program()

        # Create buffers
        ibo = self.ctx.buffer(np.ascontiguousarray(faces, dtype='i4'))
        vbo_verts = self.ctx.buffer(np.ascontiguousarray(vertices, dtype='f4'))
        vbo_attr = self.ctx.buffer(np.ascontiguousarray(attr, dtype='f4'))

        vao = self.ctx.vertex_array(
            prog,
            [
                (vbo_verts, '3f', 'i_position'),
                (vbo_attr, '3f', 'i_attr'),
            ],
            ibo,
            mode=self.mgl.TRIANGLES,
        )

        # Create framebuffer
        color_tex = self.ctx.texture((width, height), 3, dtype='f4')
        depth_tex = self.ctx.depth_texture((width, height))
        fbo = self.ctx.framebuffer(
            color_attachments=[color_tex],
            depth_attachment=depth_tex
        )

        # Render
        prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        self.ctx.clear(depth=1.0)
        self.ctx.depth_func = '<'
        self.ctx.enable(self.mgl.DEPTH_TEST)
        if cull_backface:
            self.ctx.enable(self.mgl.CULL_FACE)
        else:
            self.ctx.disable(self.mgl.CULL_FACE)

        vao.render()
        self.ctx.disable(self.mgl.DEPTH_TEST)

        # Read results
        image = np.zeros((height, width, 3), dtype='f4')
        color_tex.read_into(image)
        image = np.ascontiguousarray(image[::-1, :, :])  # Flip vertically

        depth = np.zeros((height, width), dtype='f4')
        depth_tex.read_into(depth)
        depth = np.ascontiguousarray(depth[::-1, :])

        # Cleanup
        vao.release()
        ibo.release()
        vbo_verts.release()
        vbo_attr.release()
        fbo.release()
        color_tex.release()
        depth_tex.release()

        return image, depth


def _get_gl_context():
    """Lazy-initialize the OpenGL context."""
    global _mgl_ctx
    if _mgl_ctx is None:
        _mgl_ctx = ModernGLContext()
        print("[MeshRendererGL] Created moderngl context (pure OpenGL)")
    return _mgl_ctx


def intrinsics_to_projection_np(intrinsics: np.ndarray, near: float, far: float) -> np.ndarray:
    """
    OpenCV intrinsics to OpenGL perspective matrix (numpy version).
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = np.zeros((4, 4), dtype=np.float32)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


class MeshRendererGL:
    """
    Pure OpenGL renderer for the Mesh representation.

    Uses moderngl instead of nvdiffrast for rasterization.
    This allows full resolution rendering on AMD GPUs.
    """

    def __init__(self, rendering_options={}, device='cuda'):
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1
        })
        self.rendering_options.update(rendering_options)
        self.device = device
        print(f"[MeshRendererGL] Initialized with resolution={rendering_options.get('resolution', 'default')}")

    def render(
            self,
            mesh: Mesh,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            return_types=["mask", "normal", "depth"],
            transformation=None
        ) -> edict:
        """
        Render the mesh using pure OpenGL.

        Args:
            mesh: Mesh, MeshWithVoxel, or MeshWithPbrMaterial from TRELLIS.2
            extrinsics: (4, 4) camera extrinsics
            intrinsics: (3, 3) camera intrinsics
            return_types: list of return types
            transformation: optional (4, 4) transformation matrix

        Returns:
            edict with requested render outputs
        """
        glctx = _get_gl_context()

        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]

        raster_res = resolution * ssaa

        print(f"[MeshRendererGL] Rendering: {mesh.vertices.shape[0]} verts, "
              f"{mesh.faces.shape[0]} faces, res={raster_res}px")

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            print(f"[MeshRendererGL] WARNING: Empty mesh")
            default_img = torch.zeros((3, resolution, resolution), dtype=torch.float32, device=self.device)
            return edict({k: default_img if k in ['normal', 'normal_map', 'color']
                         else default_img[:1] for k in return_types})

        # Convert to numpy
        vertices_np = mesh.vertices.cpu().numpy().astype(np.float32)
        faces_np = mesh.faces.cpu().numpy().astype(np.int32)
        extrinsics_np = extrinsics.cpu().numpy().astype(np.float32)
        intrinsics_np = intrinsics.cpu().numpy().astype(np.float32)

        # Build MVP matrix
        perspective = intrinsics_to_projection_np(intrinsics_np, near, far)
        mvp = (perspective @ extrinsics_np).astype(np.float32)

        out_dict = edict()

        for rtype in return_types:
            if rtype == "mask":
                # Render with white color, mask is where depth was written
                dummy_attr = np.ones((vertices_np.shape[0], 3), dtype=np.float32)
                _, depth = glctx.rasterize(
                    vertices_np, faces_np, dummy_attr,
                    raster_res, raster_res, mvp,
                    cull_backface=False
                )
                # Mask is where depth < 1.0 (something was rendered)
                mask = (depth < 0.9999).astype(np.float32)
                img_tensor = torch.from_numpy(mask).to(self.device)

            elif rtype == "depth":
                # Transform vertices to camera space and use Z as attribute
                vertices_homo = np.hstack([vertices_np, np.ones((vertices_np.shape[0], 1), dtype=np.float32)])
                vertices_camera = vertices_homo @ extrinsics_np.T
                # Pack camera Z into RGB (all channels same value)
                camera_z = vertices_camera[:, 2:3].astype(np.float32)
                camera_z_rgb = np.tile(camera_z, (1, 3))

                depth_img, _ = glctx.rasterize(
                    vertices_np, faces_np, camera_z_rgb,
                    raster_res, raster_res, mvp,
                    cull_backface=False
                )
                img_tensor = torch.from_numpy(depth_img[..., 0]).to(self.device)

            elif rtype == "normal":
                # Compute face normals from vertices if not available
                if hasattr(mesh, 'face_normal') and mesh.face_normal is not None:
                    face_normals = mesh.face_normal.cpu().numpy().astype(np.float32)
                else:
                    # Compute face normals from vertices
                    v0 = vertices_np[faces_np[:, 0]]
                    v1 = vertices_np[faces_np[:, 1]]
                    v2 = vertices_np[faces_np[:, 2]]
                    e0 = v1 - v0
                    e1 = v2 - v0
                    face_normals = np.cross(e0, e1)
                    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
                    face_normals = np.where(norms > 1e-10, face_normals / norms, face_normals)

                # Expand: each face gets 3 unique vertices with the face normal
                num_faces = faces_np.shape[0]
                expanded_verts = vertices_np[faces_np.flatten()]  # [F*3, 3]
                expanded_faces = np.arange(num_faces * 3, dtype=np.int32).reshape(-1, 3)
                expanded_normals = np.repeat(face_normals, 3, axis=0)  # [F*3, 3]

                normal_img, _ = glctx.rasterize(
                    expanded_verts, expanded_faces, expanded_normals,
                    raster_res, raster_res, mvp,
                    cull_backface=False
                )
                # Normalize to [0, 1] range like original
                normal_img = (normal_img + 1) / 2
                img_tensor = torch.from_numpy(normal_img).permute(2, 0, 1).to(self.device)

            elif rtype == "normal_map":
                # Vertex normal attributes (columns 3:6 of vertex_attrs)
                if hasattr(mesh, 'vertex_attrs') and mesh.vertex_attrs is not None:
                    normal_attrs = mesh.vertex_attrs[:, 3:6].cpu().numpy().astype(np.float32)
                else:
                    # Fallback: use zeros
                    normal_attrs = np.zeros((vertices_np.shape[0], 3), dtype=np.float32)

                normal_map_img, _ = glctx.rasterize(
                    vertices_np, faces_np, normal_attrs,
                    raster_res, raster_res, mvp,
                    cull_backface=False
                )
                img_tensor = torch.from_numpy(normal_map_img).permute(2, 0, 1).to(self.device)

            elif rtype == "color":
                # Vertex color attributes (columns 0:3 of vertex_attrs)
                if hasattr(mesh, 'vertex_attrs') and mesh.vertex_attrs is not None:
                    color_attrs = mesh.vertex_attrs[:, :3].cpu().numpy().astype(np.float32)
                else:
                    # Fallback: use white
                    color_attrs = np.ones((vertices_np.shape[0], 3), dtype=np.float32)

                color_img, _ = glctx.rasterize(
                    vertices_np, faces_np, color_attrs,
                    raster_res, raster_res, mvp,
                    cull_backface=False
                )
                img_tensor = torch.from_numpy(color_img).permute(2, 0, 1).to(self.device)

            else:
                raise ValueError(f"Unknown return type: {rtype}")

            # Handle SSAA downsampling
            if ssaa > 1:
                if img_tensor.dim() == 2:
                    img_tensor = img_tensor.unsqueeze(0)
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0),
                    (resolution, resolution),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                ).squeeze(0)

            out_dict[rtype] = img_tensor

        return out_dict


def create_mesh_renderer(rendering_options={}, device='cuda', backend='auto'):
    """
    Factory function to create mesh renderer with appropriate backend.

    Args:
        rendering_options: Rendering options dict
        device: PyTorch device
        backend: 'auto', 'hip', or 'gl'
            - 'auto': Use GL on AMD (higher quality), HIP on NVIDIA
            - 'hip': Force nvdiffrast HIP/CUDA backend (128px limit on AMD)
            - 'gl': Force pure OpenGL backend (slower but no resolution limit)

    Returns:
        MeshRenderer or MeshRendererGL instance
    """
    is_amd = hasattr(torch.version, 'hip') and torch.version.hip is not None

    # Environment variable override
    env_backend = os.environ.get('TRELLIS_MESH_BACKEND', '').lower()
    if env_backend in ['gl', 'opengl']:
        backend = 'gl'
    elif env_backend in ['hip', 'cuda']:
        backend = 'hip'

    use_gl = False
    if backend == 'gl':
        use_gl = True
    elif backend == 'hip':
        use_gl = False
    elif backend == 'auto':
        # On AMD, default to GL for higher quality (no 128px limit)
        if is_amd:
            use_gl = True
            print(f"[create_mesh_renderer] AMD detected -> using OpenGL backend for full resolution")
        # On NVIDIA, use HIP/CUDA for speed
        else:
            use_gl = False

    if use_gl:
        return MeshRendererGL(rendering_options, device)
    else:
        from .mesh_renderer import MeshRenderer
        return MeshRenderer(rendering_options, device)

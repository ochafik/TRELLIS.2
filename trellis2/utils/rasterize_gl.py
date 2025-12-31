"""
Pure OpenGL rasterization utilities using moderngl.

These functions provide an interface compatible with utils3d.torch.rasterize_triangle_faces
but use pure OpenGL instead of nvdiffrast, avoiding AMD HIP issues.
"""

import torch
import numpy as np

# Lazy-loaded context
_ctx = None


def _get_context():
    """Get or create the moderngl context."""
    global _ctx
    if _ctx is None:
        import moderngl
        _ctx = moderngl.create_context(standalone=True)
        print("[rasterize_gl] Created moderngl context")
    return _ctx


# Shader for rasterizing with UV coordinates
VERTEX_SHADER_UV = """
#version 330
uniform mat4 u_mvp;
in vec3 i_position;
in vec2 i_uv;
out vec2 v_uv;
out float v_depth;
void main() {
    vec4 clip_pos = u_mvp * vec4(i_position, 1.0);
    gl_Position = clip_pos;
    v_uv = i_uv;
    v_depth = clip_pos.z / clip_pos.w;
}
"""

FRAGMENT_SHADER_UV = """
#version 330
in vec2 v_uv;
in float v_depth;
layout(location = 0) out vec4 f_uv_depth;
void main() {
    f_uv_depth = vec4(v_uv, v_depth, 1.0);
}
"""

# Shader for rasterizing face IDs
VERTEX_SHADER_FACEID = """
#version 330
uniform mat4 u_mvp;
in vec3 i_position;
in float i_face_id;
out float v_face_id;
void main() {
    gl_Position = u_mvp * vec4(i_position, 1.0);
    v_face_id = i_face_id;
}
"""

FRAGMENT_SHADER_FACEID = """
#version 330
in float v_face_id;
out float f_face_id;
void main() {
    f_face_id = v_face_id;
}
"""

_programs = {}


def _get_program(ctx, name, vsh, fsh):
    """Get or create a shader program."""
    if name not in _programs:
        _programs[name] = ctx.program(vertex_shader=vsh, fragment_shader=fsh)
    return _programs[name]


def rasterize_triangle_faces_gl(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    width: int,
    height: int,
    uv: torch.Tensor = None,
    view: torch.Tensor = None,
    projection: torch.Tensor = None,
) -> dict:
    """
    Rasterize triangle faces using pure OpenGL.

    Compatible with utils3d.torch.rasterize_triangle_faces interface.

    Args:
        vertices: [B, N, 3] or [N, 3] vertex positions
        faces: [F, 3] face indices
        width, height: output resolution
        uv: [B, N, 2] or [N, 2] UV coordinates (optional)
        view: [4, 4] view matrix (optional)
        projection: [4, 4] projection matrix (optional)

    Returns:
        dict with:
            'mask': [B, H, W] bool mask
            'depth': [B, H, W] depth values
            'uv': [B, H, W, 2] UV coordinates (if uv provided)
            'face_id': [B, H, W] face IDs
    """
    ctx = _get_context()
    import moderngl

    # Handle batched input
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)
    if uv is not None and uv.dim() == 2:
        uv = uv.unsqueeze(0)

    batch_size = vertices.shape[0]
    device = vertices.device

    # Build MVP matrix
    if view is None:
        view = torch.eye(4, device=device, dtype=vertices.dtype)
    if projection is None:
        projection = torch.eye(4, device=device, dtype=vertices.dtype)
    mvp = (projection @ view).cpu().numpy().astype(np.float32)

    # Convert to numpy
    verts_np = vertices.cpu().numpy().astype(np.float32)
    faces_np = faces.cpu().numpy().astype(np.int32)

    results = {
        'mask': [],
        'depth': [],
    }
    if uv is not None:
        results['uv'] = []
        uv_np = uv.cpu().numpy().astype(np.float32)

    for b in range(batch_size):
        if uv is not None:
            # Rasterize with UV
            prog = _get_program(ctx, 'uv', VERTEX_SHADER_UV, FRAGMENT_SHADER_UV)

            vbo_verts = ctx.buffer(np.ascontiguousarray(verts_np[b]))
            vbo_uv = ctx.buffer(np.ascontiguousarray(uv_np[b]))
            ibo = ctx.buffer(np.ascontiguousarray(faces_np))

            vao = ctx.vertex_array(
                prog,
                [(vbo_verts, '3f', 'i_position'), (vbo_uv, '2f', 'i_uv')],
                ibo,
                mode=moderngl.TRIANGLES
            )

            # Create framebuffer with RGBA32F for uv+depth
            color_tex = ctx.texture((width, height), 4, dtype='f4')
            depth_tex = ctx.depth_texture((width, height))
            fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_tex)

            prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
            fbo.use()
            fbo.viewport = (0, 0, width, height)
            ctx.clear(0, 0, 0, 0, depth=1.0)
            ctx.depth_func = '<'
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.disable(moderngl.CULL_FACE)
            vao.render()

            # Read results
            uv_depth = np.zeros((height, width, 4), dtype='f4')
            color_tex.read_into(uv_depth)
            uv_depth = np.ascontiguousarray(uv_depth[::-1])

            depth_buf = np.zeros((height, width), dtype='f4')
            depth_tex.read_into(depth_buf)
            depth_buf = np.ascontiguousarray(depth_buf[::-1])

            mask = depth_buf < 0.9999
            uv_out = uv_depth[..., :2]

            results['mask'].append(torch.from_numpy(mask).to(device))
            results['depth'].append(torch.from_numpy(depth_buf).to(device))
            results['uv'].append(torch.from_numpy(uv_out).to(device))

            # Cleanup
            vao.release()
            vbo_verts.release()
            vbo_uv.release()
            ibo.release()
            fbo.release()
            color_tex.release()
            depth_tex.release()

        else:
            # Simple rasterization without UV
            prog = _get_program(ctx, 'uv', VERTEX_SHADER_UV, FRAGMENT_SHADER_UV)

            # Use dummy UVs
            dummy_uv = np.zeros((verts_np[b].shape[0], 2), dtype=np.float32)

            vbo_verts = ctx.buffer(np.ascontiguousarray(verts_np[b]))
            vbo_uv = ctx.buffer(dummy_uv)
            ibo = ctx.buffer(np.ascontiguousarray(faces_np))

            vao = ctx.vertex_array(
                prog,
                [(vbo_verts, '3f', 'i_position'), (vbo_uv, '2f', 'i_uv')],
                ibo,
                mode=moderngl.TRIANGLES
            )

            color_tex = ctx.texture((width, height), 4, dtype='f4')
            depth_tex = ctx.depth_texture((width, height))
            fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_tex)

            prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
            fbo.use()
            fbo.viewport = (0, 0, width, height)
            ctx.clear(0, 0, 0, 0, depth=1.0)
            ctx.depth_func = '<'
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.disable(moderngl.CULL_FACE)
            vao.render()

            depth_buf = np.zeros((height, width), dtype='f4')
            depth_tex.read_into(depth_buf)
            depth_buf = np.ascontiguousarray(depth_buf[::-1])

            mask = depth_buf < 0.9999

            results['mask'].append(torch.from_numpy(mask).to(device))
            results['depth'].append(torch.from_numpy(depth_buf).to(device))

            vao.release()
            vbo_verts.release()
            vbo_uv.release()
            ibo.release()
            fbo.release()
            color_tex.release()
            depth_tex.release()

    # Stack batch results
    results['mask'] = torch.stack(results['mask'])
    results['depth'] = torch.stack(results['depth'])
    if 'uv' in results:
        results['uv'] = torch.stack(results['uv'])

    return results


def rasterize_for_visibility(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    resolution: int,
    views: torch.Tensor,
    projections: torch.Tensor,
) -> torch.Tensor:
    """
    Rasterize mesh from multiple views to compute face visibility.

    Args:
        vertices: [N, 3] vertices
        faces: [F, 3] faces
        resolution: render resolution
        views: [V, 4, 4] view matrices
        projections: [V, 4, 4] projection matrices

    Returns:
        visibility: [F] int tensor with visibility counts
    """
    ctx = _get_context()
    import moderngl

    num_views = views.shape[0]
    num_faces = faces.shape[0]
    device = vertices.device

    # Expand vertices for face ID rendering
    # Each face gets 3 unique vertices with face ID as attribute
    verts_np = vertices.cpu().numpy().astype(np.float32)
    faces_np = faces.cpu().numpy().astype(np.int32)

    expanded_verts = verts_np[faces_np.flatten()]  # [F*3, 3]
    expanded_faces = np.arange(num_faces * 3, dtype=np.int32).reshape(-1, 3)
    face_ids = np.repeat(np.arange(num_faces, dtype=np.float32), 3)  # [F*3]

    prog = _get_program(ctx, 'faceid', VERTEX_SHADER_FACEID, FRAGMENT_SHADER_FACEID)

    vbo_verts = ctx.buffer(np.ascontiguousarray(expanded_verts))
    vbo_faceid = ctx.buffer(np.ascontiguousarray(face_ids))
    ibo = ctx.buffer(np.ascontiguousarray(expanded_faces))

    vao = ctx.vertex_array(
        prog,
        [(vbo_verts, '3f', 'i_position'), (vbo_faceid, '1f', 'i_face_id')],
        ibo,
        mode=moderngl.TRIANGLES
    )

    # R32F texture for face IDs
    color_tex = ctx.texture((resolution, resolution), 1, dtype='f4')
    depth_tex = ctx.depth_texture((resolution, resolution))
    fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_tex)

    visibility = torch.zeros(num_faces, dtype=torch.int32, device=device)

    for i in range(num_views):
        mvp = (projections[i] @ views[i]).cpu().numpy().astype(np.float32)

        prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        fbo.use()
        fbo.viewport = (0, 0, resolution, resolution)
        ctx.clear(-1.0, depth=1.0)  # -1 for no face
        ctx.depth_func = '<'
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        vao.render()

        # Read face IDs
        face_id_buf = np.zeros((resolution, resolution), dtype='f4')
        color_tex.read_into(face_id_buf)
        face_id_buf = np.ascontiguousarray(face_id_buf[::-1])

        # Count visible faces
        valid_ids = face_id_buf[face_id_buf >= 0].astype(np.int32)
        unique_ids = np.unique(valid_ids)
        visibility[torch.from_numpy(unique_ids).to(device)] += 1

    # Cleanup
    vao.release()
    vbo_verts.release()
    vbo_faceid.release()
    ibo.release()
    fbo.release()
    color_tex.release()
    depth_tex.release()

    return visibility

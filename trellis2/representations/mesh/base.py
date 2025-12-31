from typing import *
import os
import torch
from ..voxel import Voxel

# AMD HIP detection
_IS_AMD = hasattr(torch.version, 'hip') and torch.version.hip is not None

# Lazy imports for CUDA-only libraries
_cumesh = None
_grid_sample_3d = None


def _get_cumesh():
    """Lazy import cumesh (CUDA mesh operations)."""
    global _cumesh
    if _cumesh is None:
        try:
            import cumesh
            _cumesh = cumesh
        except ImportError as e:
            if _IS_AMD:
                print("[AMD] cumesh not available - mesh operations will use fallbacks")
            else:
                raise e
    return _cumesh


def _get_grid_sample_3d():
    """Lazy import flex_gemm grid_sample_3d with AMD fallback."""
    global _grid_sample_3d
    if _grid_sample_3d is None:
        if _IS_AMD:
            # AMD fallback: use PyTorch's native grid_sample
            def _pytorch_grid_sample_3d(attrs, coords, voxel_shape, grid, mode='trilinear'):
                """
                AMD fallback for flex_gemm's grid_sample_3d using PyTorch's grid_sample.

                This is a simplified version that assumes the voxel data can be arranged
                as a dense 3D grid for sampling.
                """
                # Convert sparse voxel to dense grid
                B, _, C = attrs.shape if len(attrs.shape) == 3 else (1, attrs.shape[0], attrs.shape[-1])
                D, H, W = voxel_shape[-3:]

                # Create dense grid
                dense_grid = torch.zeros(B, C, D, H, W, device=attrs.device, dtype=attrs.dtype)

                # Fill in sparse values (coords format: [N, 4] with batch, z, y, x)
                if coords.shape[-1] == 4:
                    batch_idx = coords[:, 0].long()
                    z_idx = coords[:, 1].long()
                    y_idx = coords[:, 2].long()
                    x_idx = coords[:, 3].long()
                else:
                    batch_idx = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
                    z_idx = coords[:, 0].long()
                    y_idx = coords[:, 1].long()
                    x_idx = coords[:, 2].long()

                # Clamp indices to valid range
                z_idx = z_idx.clamp(0, D - 1)
                y_idx = y_idx.clamp(0, H - 1)
                x_idx = x_idx.clamp(0, W - 1)

                dense_grid[batch_idx, :, z_idx, y_idx, x_idx] = attrs.reshape(-1, C)

                # Normalize grid coordinates to [-1, 1] for grid_sample
                grid_normalized = grid.clone()
                grid_normalized[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0  # x
                grid_normalized[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0  # y
                grid_normalized[..., 2] = 2.0 * grid[..., 2] / (D - 1) - 1.0  # z

                # Reshape for grid_sample: [B, 1, 1, N, 3]
                N = grid.shape[1]
                grid_5d = grid_normalized.reshape(B, 1, 1, N, 3)

                # Use PyTorch's grid_sample
                align_corners = True
                mode_str = 'bilinear' if mode == 'trilinear' else 'nearest'
                sampled = torch.nn.functional.grid_sample(
                    dense_grid, grid_5d,
                    mode=mode_str, padding_mode='zeros', align_corners=align_corners
                )

                # Reshape output: [B, C, 1, 1, N] -> [B, N, C]
                return sampled.squeeze(2).squeeze(2).permute(0, 2, 1)

            _grid_sample_3d = _pytorch_grid_sample_3d
            print("[AMD] Using PyTorch grid_sample fallback for flex_gemm")
        else:
            try:
                from flex_gemm.ops.grid_sample import grid_sample_3d
                _grid_sample_3d = grid_sample_3d
            except ImportError as e:
                raise e
    return _grid_sample_3d


def _fill_holes_trimesh(vertices, faces, max_hole_perimeter=3e-2):
    """
    AMD fallback for fill_holes using trimesh library.

    Note: trimesh's fill_holes is simpler than cumesh and may not respect
    max_hole_perimeter exactly, but it provides basic hole filling.
    """
    try:
        import trimesh
        import numpy as np

        # Convert to numpy
        verts_np = vertices.cpu().numpy()
        faces_np = faces.cpu().numpy()

        # Create trimesh mesh
        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)

        # Fill holes
        mesh.fill_holes()

        # Convert back to torch
        new_vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(vertices.device)
        new_faces = torch.from_numpy(mesh.faces.astype(np.int32)).to(faces.device)

        return new_vertices, new_faces
    except Exception as e:
        print(f"[AMD] Warning: fill_holes fallback failed: {e}")
        return vertices, faces


class Mesh:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.vertex_attrs = vertex_attrs
        
    @property
    def device(self):
        return self.vertices.device
        
    def to(self, device, non_blocking=False):
        return Mesh(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.vertex_attrs.to(device, non_blocking=non_blocking) if self.vertex_attrs is not None else None,
        )
        
    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)
        
    def cpu(self):
        return self.to('cpu')
    
    def fill_holes(self, max_hole_perimeter=3e-2):
        cumesh = _get_cumesh()

        if cumesh is not None:
            # Use CUDA cumesh (fast path for NVIDIA)
            vertices = self.vertices.cuda()
            faces = self.faces.cuda()

            mesh = cumesh.CuMesh()
            mesh.init(vertices, faces)
            mesh.get_edges()
            mesh.get_boundary_info()
            if mesh.num_boundaries == 0:
                return
            mesh.get_vertex_edge_adjacency()
            mesh.get_vertex_boundary_adjacency()
            mesh.get_manifold_boundary_adjacency()
            mesh.read_manifold_boundary_adjacency()
            mesh.get_boundary_connected_components()
            mesh.get_boundary_loops()
            if mesh.num_boundary_loops == 0:
                return
            mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
            new_vertices, new_faces = mesh.read()

            self.vertices = new_vertices.to(self.device)
            self.faces = new_faces.to(self.device)
        else:
            # AMD fallback: use trimesh
            new_vertices, new_faces = _fill_holes_trimesh(
                self.vertices, self.faces, max_hole_perimeter
            )
            self.vertices = new_vertices
            self.faces = new_faces
        
    def remove_faces(self, face_mask: torch.Tensor):
        cumesh = _get_cumesh()

        if cumesh is not None:
            vertices = self.vertices.cuda()
            faces = self.faces.cuda()

            mesh = cumesh.CuMesh()
            mesh.init(vertices, faces)
            mesh.remove_faces(face_mask)
            new_vertices, new_faces = mesh.read()

            self.vertices = new_vertices.to(self.device)
            self.faces = new_faces.to(self.device)
        else:
            # AMD fallback: simple face removal
            keep_mask = ~face_mask.cpu()
            new_faces = self.faces.cpu()[keep_mask]

            # Re-index vertices (remove unused)
            used_verts = torch.unique(new_faces)
            vertex_map = torch.zeros(self.vertices.shape[0], dtype=torch.long)
            vertex_map[used_verts] = torch.arange(len(used_verts))

            self.vertices = self.vertices[used_verts].to(self.device)
            self.faces = vertex_map[new_faces].to(self.device)

    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):
        cumesh = _get_cumesh()

        if cumesh is not None:
            vertices = self.vertices.cuda()
            faces = self.faces.cuda()

            mesh = cumesh.CuMesh()
            mesh.init(vertices, faces)
            mesh.simplify(target, verbose=verbose, options=options)
            new_vertices, new_faces = mesh.read()

            self.vertices = new_vertices.to(self.device)
            self.faces = new_faces.to(self.device)
        else:
            # AMD fallback: use trimesh simplification
            try:
                import trimesh
                import numpy as np

                verts_np = self.vertices.cpu().numpy()
                faces_np = self.faces.cpu().numpy()

                mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)

                # Use trimesh's simplify_quadric_decimation if available
                if hasattr(mesh, 'simplify_quadric_decimation'):
                    mesh = mesh.simplify_quadric_decimation(target)
                elif hasattr(mesh, 'simplify'):
                    # Older trimesh API
                    ratio = target / len(faces_np) if len(faces_np) > 0 else 1.0
                    mesh = mesh.simplify(ratio)

                self.vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(self.device)
                self.faces = torch.from_numpy(mesh.faces.astype(np.int32)).to(self.device)

                if verbose:
                    print(f"[AMD] Simplified mesh: {len(faces_np)} -> {len(mesh.faces)} faces")
            except Exception as e:
                print(f"[AMD] Warning: simplify fallback failed: {e}")


class TextureFilterMode:
    CLOSEST = 0
    LINEAR = 1


class TextureWrapMode:
    CLAMP_TO_EDGE = 0
    REPEAT = 1
    MIRRORED_REPEAT = 2


class AlphaMode:
    OPAQUE = 0
    MASK = 1
    BLEND = 2


class Texture:
    def __init__(
        self,
        image: torch.Tensor,
        filter_mode: TextureFilterMode = TextureFilterMode.LINEAR,
        wrap_mode: TextureWrapMode = TextureWrapMode.REPEAT
    ):
        self.image = image
        self.filter_mode = filter_mode
        self.wrap_mode = wrap_mode

    def to(self, device, non_blocking=False):
        return Texture(
            self.image.to(device, non_blocking=non_blocking),
            self.filter_mode,
            self.wrap_mode,
        )


class PbrMaterial:
    def __init__(
        self,
        base_color_texture: Optional[Texture] = None,
        base_color_factor: Union[torch.Tensor, List[float]] = [1.0, 1.0, 1.0],
        metallic_texture: Optional[Texture] = None,
        metallic_factor: float = 1.0,
        roughness_texture: Optional[Texture] = None,
        roughness_factor: float = 1.0,
        alpha_texture: Optional[Texture] = None,
        alpha_factor: float = 1.0,
        alpha_mode: AlphaMode = AlphaMode.OPAQUE,
        alpha_cutoff: float = 0.5,
    ):
        self.base_color_texture = base_color_texture
        self.base_color_factor = torch.tensor(base_color_factor, dtype=torch.float32)[:3]
        self.metallic_texture = metallic_texture
        self.metallic_factor = metallic_factor
        self.roughness_texture = roughness_texture
        self.roughness_factor = roughness_factor
        self.alpha_texture = alpha_texture
        self.alpha_factor = alpha_factor
        self.alpha_mode = alpha_mode
        self.alpha_cutoff = alpha_cutoff

    def to(self, device, non_blocking=False):
        return PbrMaterial(
            base_color_texture=self.base_color_texture.to(device, non_blocking=non_blocking) if self.base_color_texture is not None else None,
            base_color_factor=self.base_color_factor.to(device, non_blocking=non_blocking),
            metallic_texture=self.metallic_texture.to(device, non_blocking=non_blocking) if self.metallic_texture is not None else None,
            metallic_factor=self.metallic_factor,
            roughness_texture=self.roughness_texture.to(device, non_blocking=non_blocking) if self.roughness_texture is not None else None,
            roughness_factor=self.roughness_factor,
            alpha_texture=self.alpha_texture.to(device, non_blocking=non_blocking) if self.alpha_texture is not None else None,
            alpha_factor=self.alpha_factor,
            alpha_mode=self.alpha_mode,
            alpha_cutoff=self.alpha_cutoff,
        )


class MeshWithPbrMaterial(Mesh):
    def __init__(self,
        vertices,
        faces,
        material_ids,
        uv_coords,
        materials: List[PbrMaterial],
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.material_ids = material_ids    # [M]
        self.uv_coords = uv_coords          # [M, 3, 2]
        self.materials = materials
        self.layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }

    def to(self, device, non_blocking=False):
        return MeshWithPbrMaterial(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.material_ids.to(device, non_blocking=non_blocking),
            self.uv_coords.to(device, non_blocking=non_blocking),
            [material.to(device, non_blocking=non_blocking) for material in self.materials],
        )


class MeshWithVoxel(Mesh, Voxel):
    def __init__(self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        origin: list,
        voxel_size: float,
        coords: torch.Tensor,
        attrs: torch.Tensor,
        voxel_shape: torch.Size,
        layout: Dict = {},
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.voxel_shape = voxel_shape
        self.layout = layout

    def to(self, device, non_blocking=False):
        return MeshWithVoxel(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.origin.tolist(),
            self.voxel_size,
            self.coords.to(device, non_blocking=non_blocking),
            self.attrs.to(device, non_blocking=non_blocking),
            self.voxel_shape,
            self.layout,
        )
        
    def query_attrs(self, xyz):
        grid_sample_3d = _get_grid_sample_3d()
        grid = ((xyz - self.origin) / self.voxel_size).reshape(1, -1, 3)
        vertex_attrs = grid_sample_3d(
            self.attrs,
            torch.cat([torch.zeros_like(self.coords[..., :1]), self.coords], dim=-1),
            self.voxel_shape,
            grid,
            mode='trilinear'
        )[0]
        return vertex_attrs
        
    def query_vertex_attrs(self):
        return self.query_attrs(self.vertices)

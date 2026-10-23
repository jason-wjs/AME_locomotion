"""Convex surface pieces for the two source sparse heightfields.

Merge flat axis-aligned source quads into boxes. Preserve every remaining
nonvertical cap above the pit as a vertical convex prism, merging adjacent
coplanar triangles only when their union is convex. This keeps
the corrected source XY vertices instead of using regular-grid hfield ramps.
The caller still needs a pit floor and must verify MuJoCo contact behavior.
"""
import numpy as np
import uuid


def collision_pieces(vertices, triangles, grid_shape, pit_height=-2.0):
    grid = vertices.reshape(*grid_shape, 3)
    a, b = grid[:-1, :-1], grid[1:, :-1]
    c, d = grid[:-1, 1:], grid[1:, 1:]
    flat = ((a[:, :, 2] == b[:, :, 2]) & (a[:, :, 2] == c[:, :, 2])
            & (a[:, :, 2] == d[:, :, 2]) & (a[:, :, 2] > pit_height))
    rectangular = ((np.abs(a[:, :, 0]-c[:, :, 0]) < 1e-6)
                   & (np.abs(b[:, :, 0]-d[:, :, 0]) < 1e-6)
                   & (np.abs(a[:, :, 1]-b[:, :, 1]) < 1e-6)
                   & (np.abs(c[:, :, 1]-d[:, :, 1]) < 1e-6)
                   & (b[:, :, 0] > a[:, :, 0]+1e-6)
                   & (c[:, :, 1] > a[:, :, 1]+1e-6))
    available = flat & rectangular
    covered = available.copy()
    boxes = []
    for i in range(available.shape[0]):
        j = 0
        while j < available.shape[1]:
            if not available[i, j]:
                j += 1
                continue
            height = a[i, j, 2]
            j_end = j+1
            while (j_end < available.shape[1] and available[i, j_end]
                   and a[i, j_end, 2] == height):
                j_end += 1
            i_end = i+1
            while (i_end < available.shape[0] and available[i_end, j:j_end].all()
                   and (a[i_end, j:j_end, 2] == height).all()):
                i_end += 1
            lower = grid[i, j, :2]
            upper = grid[i_end, j_end, :2]
            boxes.append([*lower, *upper, height])
            available[i:i_end, j:j_end] = False
            j = j_end
    faces = vertices[triangles]
    u, v = faces[:, 1, :2]-faces[:, 0, :2], faces[:, 2, :2]-faces[:, 0, :2]
    twice_area = np.abs(u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0])
    nonvertical = twice_area > 1e-8
    above_pit = faces[:, :, 2].max(axis=1) > pit_height
    # Source conversion emits exactly two consecutive triangles per grid cell.
    assert len(faces) == covered.size*2
    merged = np.repeat(covered.reshape(-1), 2)
    remainder = faces[nonvertical & above_pit & ~merged]
    boxes = np.asarray(boxes)
    box_area = ((boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])).sum()
    merged_area = twice_area[merged].sum()/2
    assert np.isclose(box_area, merged_area, rtol=1e-6, atol=1e-5)
    return boxes, remainder


def merge_coplanar_caps(triangles):
    """Join connected coplanar triangles only when their union is convex."""
    triangles = np.asarray(triangles, dtype=np.float64)
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals *= np.where(normals[:, 2:3] < 0, -1.0, 1.0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    planes = np.column_stack((normals, -(normals*triangles[:, 0]).sum(axis=1)))
    edges = {}
    adjacent = [set() for _ in triangles]
    for index, triangle in enumerate(triangles):
        plane = tuple(np.round(planes[index], 6))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge = (plane, tuple(sorted((tuple(triangle[a]), tuple(triangle[b])))))
            for other in edges.get(edge, []):
                adjacent[index].add(other)
                adjacent[other].add(index)
            edges.setdefault(edge, []).append(index)
    unseen = set(range(len(triangles)))
    polygons = []
    for start in range(len(triangles)):
        if start not in unseen:
            continue
        stack, component = [start], []
        unseen.remove(start)
        while stack:
            index = stack.pop()
            component.append(index)
            for other in adjacent[index] & unseen:
                unseen.remove(other)
                stack.append(other)
        points = sorted(set(map(tuple, triangles[component].reshape(-1, 3))))

        def turn(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])

        lower, upper = [], []
        for sequence, hull in ((points, lower), (points[::-1], upper)):
            for point in sequence:
                while len(hull) >= 2 and turn(hull[-2], hull[-1], point) <= 0:
                    hull.pop()
                hull.append(point)
        hull = np.array(lower[:-1]+upper[:-1])
        area = abs(np.sum(hull[:, 0]*np.roll(hull[:, 1], -1)
                          - hull[:, 1]*np.roll(hull[:, 0], -1)))/2
        caps = triangles[component]
        u, v = caps[:, 1, :2]-caps[:, 0, :2], caps[:, 2, :2]-caps[:, 0, :2]
        source_area = np.abs(u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]).sum()/2
        planar_error = np.max(np.abs(np.asarray(points)@planes[start, :3]+planes[start, 3]))
        if np.isclose(area, source_area, rtol=1e-6, atol=1e-9) and planar_error < 1e-7:
            polygons.append(hull)
        else:
            # A concave connected patch cannot be replaced by one convex collider.
            polygons.extend(caps)
    return polygons


def add_collision_surface(spec, body, boxes, triangles, size=(8.0, 8.0), pit_height=-2.0):
    """Build convex MuJoCo geoms for the decomposed sparse source surface."""
    import mujoco

    created = []
    mesh_prefix = uuid.uuid4().hex
    bottom = pit_height-0.1
    created.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[size[0]/2, size[1]/2, pit_height-0.05],
                  size=[size[0]/2, size[1]/2, 0.05]))
    for x0, y0, x1, y1, height in boxes:
        created.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[(x0+x1)/2, (y0+y1)/2, (height+bottom)/2],
                      size=[(x1-x0)/2, (y1-y0)/2, (height-bottom)/2]))
    for index, top in enumerate(merge_coplanar_caps(triangles)):
        top = top.copy()
        u, v = top[1, :2]-top[0, :2], top[2, :2]-top[0, :2]
        if u[0]*v[1]-u[1]*v[0] < 0:
            top = top[::-1]
        lower = top.copy()
        lower[:, 2] = bottom
        count = len(top)
        faces = []
        for i in range(1, count-1):
            faces.extend((0, i, i+1, count, count+i+1, count+i))
        for i in range(count):
            j = (i+1) % count
            faces.extend((i, count+i, count+j, i, count+j, j))
        mesh = spec.add_mesh(name=f'sparse_cap_{mesh_prefix}_{index}',
                             uservert=np.concatenate((top, lower)).ravel().tolist(),
                             userface=faces)
        created.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name))
    return created

import warp as wp


collision_eps = 5.0e-3
FLT_MAX = 1.0e5
ZERO = 1.0e-6


@wp.struct
class TriangleSoup:
    mesh_id: wp.uint64
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array(dtype=int)
    edges: wp.array(dtype=int)


@wp.func
def point_triangle_distance_wp(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    v: wp.vec3,
):
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v2 - v1

    n = wp.cross(e1, e2)
    na = wp.cross(v2 - v1, v - v1)
    nb = wp.cross(v0 - v2, v - v2)
    nc = wp.cross(v1 - v0, v - v0)

    barycentric = wp.vec3(
        wp.dot(n, na), wp.dot(n, nb), wp.dot(n, nc)
    ) / wp.length_sq(n)

    bary_abs = wp.abs(barycentric)
    bary_sum = bary_abs[0] + bary_abs[1] + bary_abs[2]

    ret = float(FLT_MAX)
    distance_type = int(0)
    if bary_sum - 1.0 < ZERO:
        n_hat = wp.normalize(n)
        normal_distance = wp.dot(n_hat, v - v0)
        ret = wp.abs(normal_distance)
        distance_type = 0
    else:
        ev = v - v0
        ev3 = v - v1
        e1_hat = wp.normalize(e1)
        e2_hat = wp.normalize(e2)
        e3_hat = wp.normalize(e3)

        e1_dot = wp.dot(e1_hat, ev)
        e2_dot = wp.dot(e2_hat, ev)
        e3_dot = wp.dot(e3_hat, ev3)

        e1_norm = wp.length(e1)
        e2_norm = wp.length(e2)
        e3_norm = wp.length(e3)

        edge_distances = wp.vec3(FLT_MAX)
        if e1_norm > e1_dot > 0.0:
            p1 = v0 + e1_hat * e1_dot
            edge_distances[0] = wp.length(v - p1)
        if e2_norm > e2_dot > 0.0:
            p2 = v0 + e2_hat * e2_dot
            edge_distances[1] = wp.length(v - p2)
        if e3_norm > e3_dot > 0.0:
            p3 = v1 + e3_hat * e3_dot
            edge_distances[2] = wp.length(v - p3)

        vertex_distances = wp.vec3(
            wp.length(v - v0),
            wp.length(v - v1),
            wp.length(v - v2),
        )

        vertex_min = wp.min(vertex_distances)
        edge_min = wp.min(edge_distances)
        if vertex_min < edge_min:
            distance_type = 1
        else:
            distance_type = int(wp.argmin(edge_distances)) + 2
        ret = min(vertex_min, edge_min)

    return ret, distance_type


@wp.func
def point_projects_inside_triangle(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    v: wp.vec3,
) -> bool:
    e1 = v1 - v0
    e2 = v2 - v0
    n = wp.cross(e1, e2)
    na = wp.cross(v2 - v1, v - v1)
    nb = wp.cross(v0 - v2, v - v2)
    nc = wp.cross(v1 - v0, v - v0)

    barycentric = wp.vec3(
        wp.dot(n, na), wp.dot(n, nb), wp.dot(n, nc)
    ) / wp.length_sq(n)
    bary_abs = wp.abs(barycentric)
    bary_sum = bary_abs[0] + bary_abs[1] + bary_abs[2]
    return bary_sum - 1.0 < ZERO


@wp.func
def plane_normal(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
    return wp.normalize(wp.cross(v1 - v0, v2 - v0))


@wp.func
def inside_collision_cell(
    triangle_soup: TriangleSoup,
    neighbors: wp.array(dtype=int),
    tid: int,
    p: wp.vec3,
) -> bool:
    ret = bool(True)
    v0 = triangle_soup.vertices[triangle_soup.indices[tid * 3]]
    v1 = triangle_soup.vertices[triangle_soup.indices[tid * 3 + 1]]
    v2 = triangle_soup.vertices[triangle_soup.indices[tid * 3 + 2]]

    vertices = wp.mat33(0.0)
    vertices[0] = v0
    vertices[1] = v1
    vertices[2] = v2

    normal = plane_normal(v0, v1, v2)
    neighbor_normals = wp.mat33(0.0)
    for edge in range(3):
        neighbor = neighbors[tid * 3 + edge]
        if neighbor != -1:
            neighbor_v0 = triangle_soup.vertices[
                triangle_soup.indices[neighbor * 3]
            ]
            neighbor_v1 = triangle_soup.vertices[
                triangle_soup.indices[neighbor * 3 + 1]
            ]
            neighbor_v2 = triangle_soup.vertices[
                triangle_soup.indices[neighbor * 3 + 2]
            ]
            neighbor_normals[edge] = plane_normal(
                neighbor_v0, neighbor_v1, neighbor_v2
            )
        else:
            ret = False

    if ret:
        for edge in range(3):
            edge_normal = wp.normalize(neighbor_normals[edge] + normal)
            edge_vector = vertices[(edge + 1) % 3] - vertices[edge]
            boundary_normal = wp.normalize(wp.cross(edge_normal, edge_vector))
            if wp.dot(boundary_normal, p - vertices[edge]) < 0.0:
                ret = False
    return ret


@wp.kernel
def edge_aabbs(
    edges: wp.array(dtype=int),
    vertices: wp.array(dtype=wp.vec3),
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
):
    edge = wp.tid()
    v0 = vertices[edges[edge * 2]]
    v1 = vertices[edges[edge * 2 + 1]]
    lowers[edge] = wp.min(v0, v1)
    uppers[edge] = wp.max(v0, v1)

from collections import defaultdict
import numpy as np

def boundary_loops(mesh):
    edges = np.asarray(mesh.edges_sorted)
    if len(edges) == 0:
        return []
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        return []

    graph = defaultdict(list)
    for a,b in boundary:
        graph[int(a)].append(int(b))
        graph[int(b)].append(int(a))

    visited_edges = set()
    loops = []
    for a,b in boundary:
        edge = tuple(sorted((int(a),int(b))))
        if edge in visited_edges:
            continue
        path=[int(a),int(b)]
        visited_edges.add(edge)
        prev, cur = int(a), int(b)
        while True:
            choices=[n for n in graph[cur] if n != prev and tuple(sorted((cur,n))) not in visited_edges]
            if not choices:
                break
            nxt=choices[0]
            visited_edges.add(tuple(sorted((cur,nxt))))
            if nxt == path[0]:
                path.append(nxt)
                break
            path.append(nxt)
            prev,cur=cur,nxt
        if len(path) >= 4 and path[-1] == path[0]:
            loops.append(path)
    return loops

def suggest_margin_from_open_boundary(mesh, max_points=180):
    loops = boundary_loops(mesh)
    if not loops:
        raise ValueError("支撑体没有可检测的开放边界，请人工绘制 Margin")
    # choose loop with greatest spatial perimeter
    best = None
    best_len = -1
    for loop in loops:
        pts=np.asarray(mesh.vertices)[loop]
        length=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())
        if length > best_len:
            best_len=length; best=pts
    if len(best) > max_points:
        idx=np.linspace(0,len(best)-1,max_points,dtype=int)
        best=best[idx]
    return best

def fit_margin_plane(points):
    pts=np.asarray(points,float)
    if len(pts) < 3:
        raise ValueError("Margin 至少需要 3 个点")
    center=pts.mean(0)
    _,_,vh=np.linalg.svd(pts-center)
    normal=vh[-1]
    normal=normal/np.linalg.norm(normal)
    return center,normal

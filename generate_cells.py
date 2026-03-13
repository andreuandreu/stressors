import random
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple
from config import RADIUS, SIZE_PIX, NX_CELLS, NY_CELLS, NMERGE

@dataclass
class Cell:
    vertices: List[Tuple[int, int]]  # List of (x, y) for rectilinear polygon
    active: bool = field(default_factory=lambda: random.choice([True, False]))
    
    # computed attributes
    neighbours: List["Cell"] = field(default_factory=list, repr=False)
    radius_neighbours: List["Cell"] = field(default_factory=list, repr=False)

    @property
    def num_sides(self) -> int:
        return len(self.vertices)

    @property
    def area(self) -> int:
        # Shoelace formula for polygon area
        n = len(self.vertices)
        area = 0
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) // 2

    @property
    def center(self) -> Tuple[float, float]:
        xs = [x for x, y in self.vertices]
        ys = [y for x, y in self.vertices]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def bounding_box(self) -> Tuple[int, int, int, int]:
        xs = [x for x, y in self.vertices]
        ys = [y for x, y in self.vertices]
        return (min(xs), min(ys), max(xs), max(ys))

    def overlaps_edge_with(self, other: "Cell") -> bool:
        # Check if two rectilinear polygons share an edge
        for i in range(len(self.vertices)):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % len(self.vertices)]
            # Check horizontal edge
            if y1 == y2:
                for j in range(len(other.vertices)):
                    ox1, oy1 = other.vertices[j]
                    ox2, oy2 = other.vertices[(j + 1) % len(other.vertices)]
                    if oy1 == oy2 and y1 == oy1:
                        # Check overlap
                        if max(min(x1, x2), min(ox1, ox2)) < min(max(x1, x2), max(ox1, ox2)):
                            return True
            # Check vertical edge
            if x1 == x2:
                for j in range(len(other.vertices)):
                    ox1, oy1 = other.vertices[j]
                    ox2, oy2 = other.vertices[(j + 1) % len(other.vertices)]
                    if ox1 == ox2 and x1 == ox1:
                        # Check overlap
                        if max(min(y1, y2), min(oy1, oy2)) < min(max(y1, y2), max(oy1, oy2)):
                            return True
        return False


def split_cell(cell: Cell) -> Tuple[Cell, Cell]:
    """Split a rectangle into two rectangles with guaranteed right angles."""
    xs = [x for x, y in cell.vertices]
    ys = [y for x, y in cell.vertices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    
    if width <= 1 and height <= 1:
        return cell, cell
    
    # Decide split orientation
    if width > 1 and height > 1:
        orientation = random.choice(["h", "v"])
    elif width > 1:
        orientation = "v"
    else:
        orientation = "h"
    
    if orientation == "v":
        # Vertical split
        cut_x = min_x + random.randint(1, width - 1)
        p1 = [(min_x, min_y), (cut_x, min_y), (cut_x, max_y), (min_x, max_y)]
        p2 = [(cut_x, min_y), (max_x, min_y), (max_x, max_y), (cut_x, max_y)]
    else:
        # Horizontal split
        cut_y = min_y + random.randint(1, height - 1)
        p1 = [(min_x, min_y), (max_x, min_y), (max_x, cut_y), (min_x, cut_y)]
        p2 = [(min_x, cut_y), (max_x, cut_y), (max_x, max_y), (min_x, max_y)]
    
    return Cell(p1), Cell(p2)





def merge_cells(cells_to_merge: List[Cell]) -> Cell:
    """Union of a list of rectilinear cells returning a new Cell."""
    edges = {}
    for c in cells_to_merge:
        vs = c.vertices
        for i in range(len(vs)):
            p1 = vs[i]
            p2 = vs[(i + 1) % len(vs)]
            rev = (p2, p1)
            if rev in edges:
                del edges[rev]
            else:
                edges[(p1, p2)] = True
    # build adjacency
    adj = {}
    for (p1, p2) in edges:
        adj.setdefault(p1, []).append(p2)
    start = min(adj.keys(), key=lambda p: (p[0], p[1]))
    polygon = []
    current = start
    prev = None
    while True:
        polygon.append(current)
        nbrs = adj.get(current, [])
        nextp = None
        for n in nbrs:
            if n != prev:
                nextp = n
                break
        if nextp is None or nextp == start:
            break
        prev = current
        current = nextp
    return Cell(polygon)


def generate_cells() -> List[Cell]:
    """Create an initial NX_CELLS x NY_CELLS grid of SIZE_PIX px squares and perform exactly NMERGE operations.

    The old version used while loops that could run many times; here we simply
    execute a fixed number of merges.  Each merge picks a random starting cell
    and then grows a cluster by examining neighbouring cells until a random
    target size is reached (or no more neighbours are available).  If the
    cluster has more than one cell we replace it with its union.
    """
    # initialize grid cells
    cells: List[Cell] = []
    for i in range(NY_CELLS):
        for j in range(NX_CELLS):
            x0 = i * SIZE_PIX
            y0 = j * SIZE_PIX
            cells.append(Cell([(x0, y0), (x0 + SIZE_PIX, y0), (x0 + SIZE_PIX, y0 + SIZE_PIX), (x0, y0 + SIZE_PIX)]))

    def attempt_merge(cells: List[Cell]) -> List[Cell]:
        """Try to merge a random cluster; return updated cell list."""
        start = random.choice(cells)
        target = random.choice([2, 3, 4, 5, 6])
        cluster = [start]
        frontier = [start]
        visited_ids = {id(start)}

        # BFS using simple for-loop iteration rather than while
        for cur in frontier:
            if len(cluster) >= target:
                break
            for other in cells:
                oid = id(other)
                if oid in visited_ids:
                    continue
                if cur.overlaps_edge_with(other):
                    visited_ids.add(oid)
                    frontier.append(other)
                    cluster.append(other)
                    if len(cluster) >= target:
                        break
            if len(cluster) >= target:
                break

        if len(cluster) > 1:
            new_cell = merge_cells(cluster)
            cells = [c for c in cells if c not in cluster]
            cells.append(new_cell)
        return cells

    # perform a fixed number of merges
    merges = NMERGE
    for _ in range(merges):
        cells = attempt_merge(cells)
    return cells


def compute_neighbours(cells: List[Cell]) -> None:
    # compute direct neighbours and radius neighbours
    for i, c in enumerate(cells):
        c.neighbours.clear()
        c.radius_neighbours.clear()
    for i, c in enumerate(cells):
        for j in range(i + 1, len(cells)):
            o = cells[j]
            if c.overlaps_edge_with(o):
                c.neighbours.append(o)
                o.neighbours.append(c)
            # radius check
            dx = c.center[0] - o.center[0]
            dy = c.center[1] - o.center[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= RADIUS:
                c.radius_neighbours.append(o)
                o.radius_neighbours.append(c)


def main():
    cells = generate_cells()
    compute_neighbours(cells)
    
    # Save cells to file
    with open('cells_state_2.pkl', 'wb') as f:
        pickle.dump(cells, f)
    print(f"Saved {len(cells)} cells to cells_state_2.pkl\n")
    
    # Print summary
    for idx, c in enumerate(cells, start=1):
        print(
            f"Cell {idx}: sides={c.num_sides} area={c.area} "
            f"direct_neighbors={len(c.neighbours)} "
            f"radius_neighbors={len(c.radius_neighbours)} "
            f"active={c.active}"
        )


if __name__ == "__main__":
    main()

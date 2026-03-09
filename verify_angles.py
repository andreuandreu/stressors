import pickle
from generate_cells import Cell

def check_angle(p1, p2, p3):
    """Check angle at p2. Should be 90 or 270 degrees for rectilinear polygons."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    # For perpendicular vectors: dot product = 0
    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
    return abs(dot_product) < 0.1

with open('cells_state.pkl', 'rb') as f:
    cells = pickle.load(f)

# Check polygons of different sides
print("Angle verification for sample cells:\n")
checked = {}
for c in cells:
    s = c.num_sides
    if s not in checked:
        vertices = c.vertices
        print(f"{s}-sided cell: {vertices}")
        
        all_right_angles = True
        for i in range(len(vertices)):
            p1 = vertices[(i-1) % len(vertices)]
            p2 = vertices[i]
            p3 = vertices[(i+1) % len(vertices)]
            is_right = check_angle(p1, p2, p3)
            if not is_right:
                all_right_angles = False
                print(f"  ✗ Non-right angle at {p2}")
        
        if all_right_angles:
            print(f"  ✓ All angles are 90 degrees\n")
        else:
            print(f"  ✗ Some angles are not 90 degrees\n")
        checked[s] = True

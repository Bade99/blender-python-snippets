# Tool to automatically fix the tris that remain after using "Tris to Quads", so they become quads
import bpy
import bmesh
from math import cos, pi
import mathutils
import time
from collections import namedtuple

#TODO(fran): clearly hypotenuse calculation aint good enough.
    #Alternative 1: make the triangle a right angle rectangle and then calculate. The problem is we dont know which side should be the right angle, what we could do is execute this path only for tris whose normal is very close to an axis, and then project onto that axis's plane, then it becomes obvious which angle should be the right one. Though this looks to be prone to inconsistencies based on the shape of the object (the obj would need to be very axis aligned, which may or may not be the case depending on the object, but we could let the user decide whether to use this analyzer or not).
    #Alternative 2: analyze the two potential rectangles and measure something in them that tells us which is the right one, eg. their uniformity, their angles. And then only take the X % more likely to be correct (eg 50 % likeliest by having the greatest proportional difference in angle/uniformity/etc) and modifiy them. Then repeat the entire process.
    #INFO: One piece of important data is that we can (almost) always know which side is the short end, and therefore we always know the vertex that needs to be "moved".
    # Alternative 3: We know that most of the topology is good and follows the right gradient, and then we usually have big patches (more than 2 wrong tris close) of wrong gradient, thus we could do a gradient analysis for both alternatives and take the one that more closely follows the 'accepted' gradient.
    #  In addition to this we probably want to attack first the smaller patches of tris, because those screwed up the surrounding topology less, and also to fix first the outer most triangles, best being the ones that have no face neighbor and thus only one valid alternative to be fixed
    # Alternative 4: Yet another, more extreme solution, would be to let it all play out in both variations (always 'up' vs always 'down' (based on the tri's orientation)) and see which one requires less face changes, and keep that one.
    # Alternative 5: Finally the best option would be to execute all analyzers, maybe also weigh them based on how accurate we think they are, and pick the alternative that the mayority chose. (aka we are making an AI, said AI would be happy to have some training data that's already solved to be able compare its progress/learning)
    # Alt 7: create a huge list of affected faces from all triangles (aka the two ones at the bottom and the top of the triangle), and then analyse the gradients of the unaffected faces that connect with the top and bottom face, and choose the good side at the one that more closely follows the unaffected gradient
        #NOTE IMPORTANT: actually the best way to recover the gradient would be by looking at the shape from the outside (looking at the contour, aka the edge vertices), taking into account sections of similar facing normals and then just take the vertices that mark the whole outisde of the shape. The directions of the faces that make up the shape should match the look of the shape.
    #Alt 8: [SUPER IMPORTANT] there's a very common formation that shows up, a strip of wrong faces between two triangles, usually not longer that 4 wrong faces long, we could try to find this strips and correct them in bulk first
    
        #THIS IS CRUCIAL: for a tri to be generated there must be another tri somewhere along its face loops that it has to connect to (on 99% of cases except for some where the geometry was really screwed up, in which case you actually need to fix it before the other tri appears), there can never exist a tri alone, otherwise it couldnt be fixed, you must always thread a line to the other runaway failed tri
            #Also for starters we could check if the knifing would connect us to another triangle, and if that tri has a similar gradient then it means that we are in a tri-quad-tri strip and those three can be joined into quad-quad by knifing through the middle and dissolving both tri edges
        
    #Alt 10: start on a known good face or face region, use the gradients from that region to expand onto tri affected areas and be able to know how the final topology should look
    #NOTE IMPORTANT: tri on tri merging should only be performed if they're joined not by their short edge, and both their secondary edge gradients are very similar, comparing from the opposite starting point for each edge to the other point on each edge (otherwise we'd end up with isosceles looking quads instead of rectangular ones)
        #Extra (IMPORTANT?): this avoids creating the famous 'diamond' quad, which is actually a very much used and known shape for a quad, but it is extremely uncommon when compared with rectangular quads. Thus I should probably take it into account, but only generate one when there's nothing left to do, aka in the last iteration of the analyser (after having converted everything else that it could)
    #NOTE: bpy.ops.mesh.select_more() #Select More (Ctrl + Numpad +): selects shared faces. Could be useful for quickly finding the non damaged areas of the mesh
    #TODO(fran): I would like to know if there are situations where the indices in the mesh and the ones in bmesh could become disjointed, or when removing verts/edges/faces if that invalidates other indices
    #SUPER IMPORTANT NEW DISCOVERY: the tri-quads-tri strip has an even better property, when in face mode, if pressing Alt on the face to follow the face loop we (almost) always arrive at its brother tri, we can actually simply fix the entire strip in one go. The one thing we need to solve is which side to take, but that can probably be always found out by a combination of attributes (tri strip length, normals, rectangle shapes, and so on)
#INFO SUPER USEFUL: bpy.ops.mesh.vert_connect() (aka vertex connect) is MUCH faster than using the knife tool!
    #INFO: Following with the tri strip idea, a good way to start would be to find the tris closest to the edges of the mesh, though this would only work if the mesh was open, for totally closed meshes we cant find any starting point
    #INFO: there area areas so screwed up that the two tris from the strip dont connect directly along their face loop, because at some point a diamond shape send each one onto opposite directions, in this case, where both tris cant find a match, we would need to find the closest tri-strip match, we can do this because both tris will share the last face before going their separate ways, therefore we'd need to connect both strips through that face

def print(data):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CONSOLE':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT")      

def objectmode(): bpy.ops.object.mode_set(mode = 'OBJECT')

def editmode(): bpy.ops.object.mode_set(mode = 'EDIT')

def updatesel():  #To update the current selection made in Edit mode you need to toggle between Object and back to Edit (visually the selection looks up to date but when querying it via via python it is outdated)
    objectmode()
    editmode()

def print_tri_err_list(ls, msg):
    cnt = len(ls)
    if cnt > 0:
        print(msg)
        if isinstance(ls, set):
            cnt = sum(len(x) for x in ls)
        print(f"For {cnt} Tri(s) of index: {ls}")
        print("")
    return cnt

def fmtVec3f(v): return f"<Vector ({v.x:.6f}, {v.y:.6f}, {v.z:.6f})>"

def flatten(xss): return [x for xs in xss for x in xs]

def print_and_clear_tri_errors(errs_to_print):
    print(":=================================:")
    print("Error Report:")
    print("(To quickly find a Triangle referenced here deselect any selected faces, go to Object Mode, open the Python Console and type bpy.context.active_object.data.polygons[INDEX OF THE TRIANGLE].select=True then go back to edit mode)")
    print("")
    err_cnt = 0
    for e in errs_to_print:
        err_cnt += print_tri_err_list(*e)
        e[0].clear()
    print(":=================================:")
    return err_cnt

def get_face_edges_sorted_by_length(f): return sorted(f.edges, reverse= True, key = lambda e: (e.verts[0].co - e.verts[1].co).length_squared)

def deg_to_rad(angle_deg): return angle_deg*pi/180

def get_face_angles(f):
    edgepairs = [[e for e in f.edges if v in e.verts] for v in f.verts]
    assert all(len(ep) == 2 for ep in edgepairs), "I screwed something up in the edge pair calculation"
    angles = []
    for ep in edgepairs:
        (shared_vert,) = set(ep[0].verts).intersection(set(ep[1].verts)) #Raises error if set doesnt contain only one element
        (v2,v3,) = set(ep[0].verts).symmetric_difference(set(ep[1].verts))
        angle = (v2.co - shared_vert.co).angle(v3.co - shared_vert.co)
        assert angle <= pi+1e-07
        angles.append(angle)
    return angles

def isConvex(points): #Note: the order of the points matters as it determines the shape of the polygons that is analysed
    n = len(points) #nº edges
    prev = 0 # Stores direction of cross product of previous traversed edges
    curr = 0 # Stores direction of cross product of current traversed edges
    for i in range(n):
        temp = [points[i-2], points[i-1], points[i]]
        curr = (temp[1] - temp[0]).cross(temp[2] - temp[0])
        if (curr != 0):
            if (curr * prev < 0): return False # If direction of cross product of all adjacent edges are not same
            else: prev = curr
    return True

def project2D(points, face): #Projects the 3D points (mathutils.Vector) in the array onto the face returning the corresponding array of 2D points on that plane
    basis1 = (face.verts[0].co - face.verts[1].co).normalized()
    basis2 = face.normal.cross(basis1).normalized()
    origin = face.verts[0].co
    #x = (p - origin).dot(basis1)
    #y = (p - origin).dot(basis2)
    return [mathutils.Vector(((v - origin).dot(basis1), (v - origin).dot(basis2))) for v in points]

def get_ordered_vertices(face):
    assert len(face.verts) == 4
    v1 = face.verts[0]
    connected_edges = [e for e in face.edges if v1 in e.verts]
    assert len(connected_edges) == 2
    connected_verts = [v for v in flatten([e.verts for e in connected_edges]) if v != v1]
    assert len(connected_verts) == 2
    v2 = connected_verts[0]
    v4 = connected_verts[1]
    (v3,) = [v for v in face.verts if v not in [v1, v2, v4]]
    return [v1, v2, v3, v4]

editmode()
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='FACE')
bpy.ops.mesh.select_face_by_sides(number=3, extend=True)
updatesel()

#INFO: bpy.ops.mesh needs to run while in Edit mode, but edge.select (aka bpy.context) needs to run in Object mode
#WARNING: going back and forth between Object and Edit mode frees the mesh arrays, previous data gets corrupted, therefore, when transitioning I must convert objects to indices to be able to refer back to them later


mesh = bpy.context.active_object.data

bm = bmesh.from_edit_mesh(mesh)

normal_val_threshold_angle = 10 #degrees
normal_val_threshold = cos(normal_val_threshold_angle*pi/180)

err_equilateral_isosceles, err_equilateral_isosceles_msg = [], "Error! Triangle is equilateral or isosceles, not quite sure what to do with it, skipping it..."
err_zero_length, err_zero_length_msg = [], "Error! Triangle has a zero length edge, you should check that out, I'm skipping it..."
err_no_candidate, err_no_candidate_msg  = [], "Error! Couldn't find any valid candidate faces adjacent to the Triangle, that's strange (probably surrounded by more incorrect up topology), I'm skipping it..."
err_too_many_candidates, err_too_many_candidates_msg = [], "Error! There's more than two valid candidate faces adjacent to the Triangle, that's strange (probably some vertices overlapping), maybe I could solve it anyways, but for the moment I'm skipping it..."
err_one_candidate, err_one_candidate_msg = [], "Error! There's only one valid candidate face adjacent to the Triangle, I don't yet handle this case, so it's a TODO, but for the moment I'm skipping it..."
err_faces_too_similar, err_faces_too_similar_msg = [], "Error! Both knife candidate faces are too similar for the analyser to decide for this Triangle, you deal with it, I'm skipping it..."
err_normal_threshold, err_normal_threshold_msg = [], f"Error! The best candidate face has a normal more than {normal_val_threshold_angle}º dissimilar from that on the Triangle, there's a high likelihood that this was actually a real triangle in the original mesh, I'm skipping it..."
err_short_edge, err_short_edge_msg = [], "Error! The best candidate triangle is linked to us by its short edge, this could be a quad with a diamond shape, but since those are very uncommon it's more likely to simply not be the correct match, this is a TODO, I'll look at it later, but for the moment I'm skipping it..."
err_shape_not_rectangular, err_shape_not_rectangular_msg = [], "Error! Merging with the best candidate triangle would form a shape too dissimilar from a rectangle, I'm skipping it..."
err_multiple_dissolve, err_multiple_dissolve_msg = set(), "Error! Performing the combined edge dissolve would generate a face with more than 4 vertices, this is a TODO, but for the moment I'm skipping it..."
err_not_convex, err_not_convex_msg = [], "Error! Neither candidate face would generate a convex quad (this may be a TODO indicating we have to analyse the tri's short edge or it may indicate that the area around the tri is still too damaged), but for the moment I'm skipping it..."

errs_to_print = [(err_equilateral_isosceles,err_equilateral_isosceles_msg), (err_zero_length,err_zero_length_msg), (err_no_candidate,err_no_candidate_msg), (err_too_many_candidates,err_too_many_candidates_msg), (err_one_candidate,err_one_candidate_msg), (err_faces_too_similar,err_faces_too_similar_msg), (err_normal_threshold,err_normal_threshold_msg), (err_short_edge, err_short_edge_msg), (err_shape_not_rectangular, err_shape_not_rectangular_msg), (err_multiple_dissolve, err_multiple_dissolve_msg), (err_not_convex, err_not_convex_msg)]


tris = [f for f in bm.faces if f.select and len(f.verts) == 3] #Find all selected triangles
#TODO(fran): accept a user selected area as working area, aka the user selects a region and we only take the tris from there and only fix those

if False: #Testing
    bm.faces.ensure_lookup_table() #so we can reference faces directly via bmesh.faces[idx]
    tris = [bm.faces[254]]

t = True
f = False

tritri = f
triquad = not tritri

do_cleanup = t
visualization = not do_cleanup

debug_info = f

print(":xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx:")
print(f"Initial triangle count: {len(tris)}")

#TODO(fran): before doing anything alert about faces with more than 4 verts

if tritri: #Tri-Tri Analysis
    print("--------------Tri-Tri Analysis--------------")
    
    linked_faces_idx = set() #before transitioning into object mode we must convert the objects into their indices to still be able to reference them

    LinkedFace = namedtuple('LinkedFace', 'edge face n_val')

    connected_tris = 0 #TODO(fran): refine the calcuation for this number

    for tri in tris:
        
        if debug_info: print(f"Triangle index: {tri.index}")
        
        sorted_tri_edges = get_face_edges_sorted_by_length(tri)
        #IMPORTANT INFO: idx 4590 (in the front grill) is an interesting case of (apparently) almost isosceles, the short edge is definitely not the one I would expect just by looking at it. So this would be worth considering for establishing a threshold for tris categorized as isosceles, in which case we could ignore them or analyse all 3 sides
        #TODO(fran): the short edge discrimination should only be done if it is clearly shorter (eg more than 30% shorter than the next one), otherwise we are sometimes skipping the actually correct edge that we should merge with
        
        #TODO(fran): this is horrible and impossible to read or understand
        linked_faces = []
        for e in sorted_tri_edges[:-1]: #iterate over all edges but the short one
            f = next((lf for lf in e.link_faces if lf != tri and len(lf.verts) == 3), None)
            if f:
                linked_faces += [LinkedFace(e, f, tri.normal.dot(f.normal))]  #(edge connecting the two faces, face (tri) to merge, normal dot product value)
        linked_faces.sort(reverse= True, key = lambda n: n.n_val)
        
        if linked_faces:
            connected_tris += 1
            if linked_faces[0].n_val < normal_val_threshold:
                err_normal_threshold.append(tri.index)
                continue
            if get_face_edges_sorted_by_length(linked_faces[0].face)[2] == linked_faces[0].edge: #TODO(fran): this connections should be saved and analysed at the end for possible valid diamond shapes
                err_short_edge.append((tri.index,linked_faces[0].face.index))
                #TODO(fran): it may be a good idea to then check if linked_faces has any other alternative that we want to match
                continue
            
            edgeteams = []
            edgeteams.append([e for e in tri.edges if e != linked_faces[0].edge])
            assert len(edgeteams[0])==2
            edgeteams.append([e for e in linked_faces[0].face.edges if e != linked_faces[0].edge])
            assert len(edgeteams[1])==2
            edgeteams.append([edgeteams[0][0], next(e for e in edgeteams[1] if any(v in e.verts for v in edgeteams[0][0].verts))])
            assert len(edgeteams[2])==2
            edgeteams.append([edgeteams[0][1], next(e for e in edgeteams[1] if any(v in e.verts for v in edgeteams[0][1].verts))])
            assert len(edgeteams[3])==2
            
            #check if generated shape is not rectangular:
            rectangle_threshold_angle = 90 #degrees
            rectangle_threshold_angle_epsilon = 53 #degrees plus minus #TODO(fran): needs more refinement (or just make it user settable)
            rad_min = deg_to_rad(rectangle_threshold_angle - rectangle_threshold_angle_epsilon)
            rad_max = deg_to_rad(rectangle_threshold_angle + rectangle_threshold_angle_epsilon)
            #TODO(fran): I dont have a lot of trust on this angle analysis, I'd prefer to do gradient analysis between opposing sides of the generated quad
            rect_shape_fail = False
            for et in edgeteams:
                angle = (et[0].verts[0].co-et[0].verts[1].co).angle(et[1].verts[0].co-et[1].verts[1].co)
                if debug_info: print(f"Angle Min: {rectangle_threshold_angle-rectangle_threshold_angle_epsilon} | Angle Max: {rectangle_threshold_angle+rectangle_threshold_angle_epsilon} | Current Angle: {angle*180/pi}")
                if not rad_min <= angle <= rad_max:
                    err_shape_not_rectangular.append((tri.index,linked_faces[0].face.index))
                    rect_shape_fail = True
                    break
            if rect_shape_fail:
                continue
            
            #TODO(fran): avoid making modifications in areas categorized as 'too messy', for example avoid faces attached to vertices that have more than 4 or 5 edges connected to them (this would avoid, for example, triangle fans)
            
            #TODO(fran): additional checks to make sure the main candidate triangle should be merged with us or not
            if len(linked_faces) > 1:
                if linked_faces[0].n_val == linked_faces[1].n_val:
                #if abs(linked_normals[0].n_val - linked_normals[1].n_val) < 1e-07:
                    err_faces_too_similar.append(tri.index)
                    continue
            linked_faces_idx.add((linked_faces[0].edge.index,*sorted([tri.index, linked_faces[0].face.index]))) #TODO(fran): instead of using a set to remove duplicates we should look for a duplicate, in which case it makes for a stronger case that those two tris should be merged
    
    
    multi_faces_to_remove = []
    for i,fi in enumerate(linked_faces_idx): #TODO(fran): optimize this search
        multi_faces = [(i2,fi2) for i2, fi2 in enumerate(linked_faces_idx) if any(f in [fi[1],fi[2]] for f in [fi2[1],fi2[2]])] #TODO(fran): what I originally wanted to do was to check if they shared an additional edge different from the vertex that connects them, but checkin for faces was easier, though it may introduce some unexpected bugs  #TODO(fran): this could be calculated while inside the main loop the reduce the performance cost by having the first tris iterate over less elements, though idk if that would affect other selections, it may be best to let it all play out first and then decide if something is wrong, idk

        if len(multi_faces) > 1: #there's always at least one because we include ourselves (aka the current linked_face)
            
            err_multiple_dissolve.add(tuple(sorted(list(set([*flatten([[i2[1][1],i2[1][2]] for i2 in multi_faces])]))))) #(this is so dumb)
            multi_faces_to_remove.append(fi) #NOTE(fran): after iterating over everything we remove them all at once (otherwise there would be cases where we could be missing some additional multi_faces simply because of the fact that we already removed the other edge)
    
    for f in multi_faces_to_remove: linked_faces_idx.remove(f)
    
    
    bpy.ops.mesh.select_all(action='DESELECT')
    
    if visualization:
        #bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='FACE')
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='EDGE')
        #bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT')

        objectmode()
    
        for fi in linked_faces_idx:
            mesh.edges[fi[0]].select = True
            #mesh.polygons[fi[1]].select = True
            #mesh.polygons[fi[2]].select = True

        editmode()
        updatesel()
    
    elif do_cleanup:
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='EDGE')
        objectmode()
        for fi in linked_faces_idx: mesh.edges[fi[0]].select = True
        editmode()
        updatesel()
        bpy.ops.mesh.dissolve_edges()
        #TODO(fran): all this quads that we generated from tris should be saved for later, it'd be useful to check if a quad that a tri wants to knife was generated from 2 tris, in which case it is more than likely the correct quad and the tri should try to knife someone else

    cnt_fixed = len(linked_faces_idx)*2
    print(f"Connected tris analysed: {connected_tris}")
    cnt_skipped = print_and_clear_tri_errors(errs_to_print)
    print(f"Merged {cnt_fixed} triangles into {cnt_fixed//2} quads!")
    print(f"Skipped {cnt_skipped} triangles.")


elif triquad: #Tri-Quad Analysis
    print("--------------Tri-Quad Analysis--------------")
    
    final_faces = [] #array of tuples [(vertex_to_move_index, tri_edge_index, face_to_knife_index, v1_index_connect_from, v2_index_connect_to),...]
    cnt_fixed = 0
    
    #TODO(fran): before doing all this mess call the triangle-on-triangle routine, aka check for tris that are next to each other and thus can simply be merged together into a correct quad

    for tri in tris:
        sorted_tri_edges = get_face_edges_sorted_by_length(tri) # idx 0 is the longest edge, idx 1 the middle one, idx 2 for sure the shortest one
        
        edgelengths = [(e.verts[0].co - e.verts[1].co).length_squared for e in sorted_tri_edges]
        if edgelengths[0] == edgelengths[1] or edgelengths[1] == edgelengths[2]:
            err_equilateral_isosceles.append(tri.index)
            continue
        elif edgelengths[0] == 0.0 or edgelengths[1] == 0.0 or edgelengths[2] == 0.0:
            err_zero_length.append(tri.index)
            continue
        
        short_edge = sorted_tri_edges[2]
        
        vertex_to_move = next(v for v in tri.verts if v not in short_edge.verts)
        
        final_face = ()
        
        LinkedCandidate = namedtuple('LinkedCandidate', 'vert edge face n_val vert_knife_from, vert_knife_to convexity') #reminder old:(vertex_to_move_idx, sorted_tri_edge_indices[1], f.index)
        linked_faces = []
        for e in sorted_tri_edges[:-1]: #iterate over all edges but the short one
            f = next((lf for lf in e.link_faces if lf != tri and len(lf.verts) == 4), None)
            if f:
                linked_faces += [LinkedCandidate(vertex_to_move, e, f, 0, 0, 0, 0)]  #(vertex to move, edge connecting the two faces, face (tri) to merge, normal dot product value, ...)
        
        #TODO(fran): add the second possible connection point (aka the other possible 'vertex_to_move') into the candidate face (also compare normal and shape)
        
        if len(linked_faces) == 2:
            #Calculate the generated normal for both candidates and choose the one closest to the original, aka the one that makes the face the most planar/coplanar
            #TODO: find out the correct winding order for calculating the normals so we can avoid doing two dot products
            
            #TODO(fran): given float precision is limited, we want to calculate the cosine of the angle between the two normals WITHOUT adding to the formula the multiplication by the modulus of the two normals, given that their modulus will be very close by usually not exactly 1 and could be enough to screw up the analysis. Though here again the question would be, how accurate is the cosine function? Cause it could actually be less precise, I doubt it would change the winner, but it could create more ties
            
            shares_edge_with = lambda v1, v2, f: any([all(v in [v1, v2] for v in verts) for verts in [e.verts for e in f.edges]]) #vertices v1 and v2 share an edge from face f #BUG IMPORTANT: should check that vi1 != vi2 but for performance im gonna ignore it here and remove those elements from the list before calling this function
            
            
            debug_normals = debug_info
            if debug_normals:
                print(f"Triangle index: {tri.index}")
                print(f"Triangle Normal: {tri.normal}")
                print(f"Vrtx to move index: {linked_faces[0].vert.index}") #NOTE: for now this is constant no matter which edge we're analysing, BEWARE this will need to be parameterized once we accept multiple knifing_to possibilities
            
            for i, lf in enumerate(linked_faces):
                #Calculate the candidate knifed tri's normal
                secondary_vert = next(v for v in lf.edge.verts if v != lf.vert)
                p1 = lf.edge.verts[0].co
                p2 = lf.edge.verts[1].co
                p3_vert = next(v for v in lf.face.verts if v != lf.vert and v != secondary_vert and shares_edge_with(v, lf.vert, lf.face))
                p3 = p3_vert.co
                n1 = ((p2 - p1).cross(p3 - p1)).normalized() #TODO(fran): check the cross product isnt 0 (indicates a bug somewhere)
                n2 = -n1
                #Note: since idk the correct winding order and thus neither the direction of the normal, I'll simply try both directions and keep the one that best matches, since, no matter what, normals from faces close together should always be pretty similar (dot product closer to 1 -> more similar normals)
                n1_val = tri.normal.dot(n1)
                n2_val = tri.normal.dot(n2)
                n_val = max(n1_val, n2_val)
                
                (tri_vert,) = [v for v in tri.verts if v not in lf.edge.verts]
                ordered_points = [p1, tri_vert.co, p2, p3]
                convexity = isConvex(project2D(ordered_points, tri)) #TODO(fran): im not quite sure onto which face the projection should be made (the tri's or the possibly soon to be newly generated quad's)
                
                linked_faces[i] = LinkedCandidate(linked_faces[i].vert, linked_faces[i].edge, linked_faces[i].face, n_val, secondary_vert, p3_vert, convexity) #TODO(fran): this is ugly, I have to do it this way because tuples cant be modified, it may be better to use a dataclass
                
                if debug_normals:
                    print(f"Candidate {i} face index: {lf.face.index}")
                    print(f"Vrtx secondary index: {secondary_vert.index}")
                    print(f"Vertex Indices to merge with tri: {lf.edge.verts[0].index}, {lf.edge.verts[1].index}, {p3_vert.index}")
                    print(f"C{i} Point1: " + fmtVec3f(p1))
                    print(f"C{i} Point2: " + fmtVec3f(p2))
                    print(f"C{i} Point3: " + fmtVec3f(p3))
                    print(f"C{i} Vec1: " + fmtVec3f(p2 - p1))
                    print(f"C{i} Vec2: " + fmtVec3f(p3 - p1))
                    print(f"C{i} N1: " + fmtVec3f(n1))
                    print(f"C{i} N2: " + fmtVec3f(n2))
                    print((f"C{i} N1 won with val: " if n_val == n1_val else f"C{i} N2 won with val: ") + str(n_val))
            
            linked_faces.sort(reverse= True, key = lambda n: n.n_val)
            
            if linked_faces[0].convexity and linked_faces[1].convexity:
                if abs(linked_faces[0].n_val - linked_faces[1].n_val) < 1e-08:
                #if abs(linked_faces[0].n_val - linked_faces[1].n_val) < cos(.5*pi/180): #TODO(fran): This basically errors for every tri, we dont have precision even to .5º, the normal analyser is starting to look like a 50-50 coin toss
                    err_faces_too_similar.append(tri.index)
                    continue
                elif linked_faces[0].n_val < normal_val_threshold:
                    err_normal_threshold.append(tri.index)
                    continue
            elif linked_faces[0].convexity:
                if linked_faces[0].n_val < normal_val_threshold:
                    err_normal_threshold.append(tri.index)
                    continue
            elif linked_faces[1].convexity:
                if linked_faces[1].n_val < normal_val_threshold:
                    err_normal_threshold.append(tri.index)
                    continue
                linked_faces[0], linked_faces[1] = linked_faces[1], linked_faces[0]
            else:
                err_not_convex.append(tri.index)
                continue
            #TODO(fran): face angles check for err_rectangular_shape
            #TODO(fran): add the err_multiple_dissolve case from tri-tri to avoid generating n-gons
            
            final_face = (linked_faces[0].vert.index, linked_faces[0].edge.index, linked_faces[0].face.index, linked_faces[0].vert_knife_from.index, linked_faces[0].vert_knife_to.index) #(tri vertex linked to its 2 longest edges, edge linked to the face to be knifed, face to be knifed, tri secondary vertex aka knife_from, quad to be knifed's vertex aka knife_to )
            
            #IMPORTANT TODO(fran): we should consider when the values obtained here are significant, by adding an epsilon that has to separate the two normals, because as we know, if the surface itself is very flat there aint much this analyser can do. Therefore for example we should consider the result to be significant if there's more than a 3º difference between normals (for example)
            
            #INFO IMPORTANT: the 'normal' analyser will not work well with flat surfaces, ie it works best the more curved the surface is, at least in that case we will for sure need to use a different analyser, good thing is that analyser at least already has the info that the surface is very flat. (Update: with the meshes i've tested so far it's been working great in all scenarios, we may not need another analyser!)
            
            #TODO(fran): there's a case that's not being handled, and that is to move the vertex_to_move one edge further. Sometimes a diamond face that cuts through two faces loops shows up, and even if it has a tri at the top and bottom neither can cut it because they actually dont want to cut to the following edge, but to the next one over (I left a screenshot in my Windows Screenshots folder that illustrates what im trying to say). It can also end up generating all quads but in the completely wrong shape (I left another Screenshot with this, it's actually worse than can be seen in the picture because there are overlapping edges), though I think this case will be fixed by itself when we add n-gon generation prevention (following up with this, there seems to be cases stemming from this where strange things start to happen, like verts getting totally misplaced or gone missing)
            
        elif len(linked_faces) > 2:
            err_too_many_candidates.append(tri.index)
            continue
        elif len(linked_faces) == 1: #TODO(fran): I dont currently want to handle this case since it probably means there's a lot of crap geometry around this tri. I would though want to handle it for some cases and DEFINITELY for the case where there's actually no face connected to it, which means it's the last face of the face loop and can be handled as usual. This case should be merged with the '2' case.
            err_one_candidate.append(tri.index)
            continue
        elif len(linked_faces) == 0:
            err_no_candidate.append(tri.index)
            continue
        
        final_faces.append(final_face) #add the best candidate for knifing
        
        cnt_fixed +=1

    
    bpy.ops.mesh.select_all(action='DESELECT')

    #bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT')
    #bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='EDGE')
    bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='FACE')
    objectmode()
    if visualization:
        for ff in final_faces:
            #mesh.edges[ff[1]].select = True
            mesh.polygons[ff[2]].select = True

    if do_cleanup:
        editmode()
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT')
        for ff in final_faces: #TODO(fran): all the context changes make this loop very slow. Optimizations: connect multiple tris at a time (make sure they are not connected/near other selected tris), or directly operate over the bmesh (with bmesh.ops and bmesh.utils) and then update the real mesh
                #NOTE: the fail case occurs when the selected vertices make a face be fully selected (that's when some vertices dont get connected/paired), we should make groups of non face making vertices and select them all at once so this loop gets a lot faster
                    #Another option would be to select everything, connect it, then go back and test that a new edge was made between each vertex pair, if it was we remove it from the list, otherwise we keep it and run the loop again
            objectmode()
            mesh.vertices[ff[3]].select = True
            mesh.vertices[ff[4]].select = True
            editmode()
            updatesel()
            bpy.ops.mesh.vert_connect()
            bpy.ops.mesh.select_all(action='DESELECT')
        
        editmode()
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='EDGE')
        objectmode()
        for ff in final_faces: mesh.edges[ff[1]].select = True
        editmode()
        updatesel()
        bpy.ops.mesh.dissolve_edges()
        
    #TODO?: notify when we knife the same face twice

    editmode()
    updatesel()

    cnt_skipped = print_and_clear_tri_errors(errs_to_print)
    print(f"Cleaned up {cnt_fixed} triangles!")
    print(f"Skipped {cnt_skipped} triangles.")


print(":xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx:")
print("")
#t0 = time.perf_counter()
#print(str((time.perf_counter()-t0)*1000) + "ms")
bm.free()
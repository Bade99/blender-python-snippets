import bpy
import bmesh
from collections import namedtuple
from math import pi
import mathutils

registering = False

bl_info = {
    "name" : "Tris to Quads Improved",
    "author": "Mima",
    "version": (0, 1),
    "blender": (2, 80, 0),
    "location": "View3d > Sidebar",
    "description": "Tool to correct mistakes made by the default Tris to Quads functionality",
    "category": "Face"
}

class FixSelectedQuads(bpy.types.Panel):
    bl_label = "Tris To Quads Improved"
    bl_idname = "PT_TrisToQuadsImproved_FixSelectedQuads"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tris To Quads"
    
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator(FixSelectedQuadsOp.bl_idname, text="Fix Selected Quads")

class FixSelectedQuadsOp(bpy.types.Operator):
    bl_idname = "mesh.fix_selected_quads"
    bl_label = "Fix Selected Quads Op"

    def execute(self, context):
        FixQuads(True, self.report)
        return {'FINISHED'}

classes = [FixSelectedQuads, FixSelectedQuadsOp]
def register(): 
    for c in classes: bpy.utils.register_class(c)
def unregister():
    for c in classes: bpy.utils.unregister_class(c)
if __name__ == "__main__": register()


def print(data):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CONSOLE':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT")

def objectmode(): bpy.ops.object.mode_set(mode = 'OBJECT')

def editmode(): bpy.ops.object.mode_set(mode = 'EDIT')

def updatesel():
    objectmode()
    editmode()

def flatten(xss): return [x for xs in xss for x in xs]

def link_faces(face): return [f for f in flatten([e.link_faces for e in face.edges]) if f != face]

def isTri(face): return len(face.verts) == 3

def isQuad(face): return len(face.verts) == 4

def shares_edge_with(v1, v2, f): return any([all(v in [v1, v2] for v in verts) for verts in [e.verts for e in f.edges]]) or v1 == v2 #vertices v1 and v2 share an edge from face f #INFO: for what I require in this code I need that on the case of v1 == v2 it also returns true

def get_face_angles(f):
    edgepairs = [[e for e in f.edges if v in e.verts] for v in f.verts]
    assert len(edgepairs) == 4, "This is not a quad"
    assert all(len(ep) == 2 for ep in edgepairs), "I screwed something up in the edge pair calculation"
    angles = []
    for ep in edgepairs:
        (shared_vert,) = set(ep[0].verts).intersection(set(ep[1].verts)) #Raises error if set doesnt contain only one element
        (v2,v3,) = set(ep[0].verts).symmetric_difference(set(ep[1].verts))
        angle = (v2.co - shared_vert.co).angle(v3.co - shared_vert.co)
        assert angle <= pi+1e-07
        angles.append(angle)
    return angles

def get_face_angles_from_points(points): #Calculates the angles based on the order of the points
    angles = []
    for i in range(len(points)):
        p1, p2, p3 = points[i-2], points[i-1], points[i]
        angles.append((p1-p2).angle(p3-p2))
    return angles
    
    edgepairs = [[e for e in f.edges if v in e.verts] for v in f.verts]
    assert len(edgepairs) == 4, "This is not a quad"
    assert all(len(ep) == 2 for ep in edgepairs), "I screwed something up in the edge pair calculation"
    angles = []
    for ep in edgepairs:
        (shared_vert,) = set(ep[0].verts).intersection(set(ep[1].verts)) #Raises error if set doesnt contain only one element
        (v2,v3,) = set(ep[0].verts).symmetric_difference(set(ep[1].verts))
        angle = (v2.co - shared_vert.co).angle(v3.co - shared_vert.co)
        assert angle <= pi+1e-07
        angles.append(angle)
    return angles

def isConvex(points): #Note: the order of the points matters as it determines the shape of the polygon that is analysed
    n = len(points) #nº edges
    prev = 0 # Stores direction of cross product of previous traversed edges
    curr = 0 # Stores direction of cross product of current traversed edges
    
    for i in range(n):
        temp = [points[i-2], points[i-1], points[i]]
        curr = (temp[1] - temp[0]).cross(temp[2] - temp[0]) #TODO(fran): shouldnt temp[1] be the one both vectors subtract against? Im confused as to why this works
        if (curr != 0):
            if (curr * prev < 0): return False # If direction of cross product of all adjacent edges are not same
            else: prev = curr
    return True

def findConcavityVertexIdx(points): #Returns the index into the verts array where the concavity point appears (only works for quads) #Note: the order of the points matters as it determines the shape of the polygon that is analysed
    n = len(points) #nº edges
    assert n == 4
    dirs = []
    for i in range(n):
        temp = [points[i-2], points[i-1], points[i]]
        dirs.append((temp[1] - temp[0]).cross(temp[2] - temp[0]))

    positives = [x for x in dirs if x > 0]
    negatives = [x for x in dirs if x < 0]
    pos_cnt = len(positives)
    neg_cnt = len(negatives)
    print(dirs)
    assert pos_cnt != neg_cnt 
    
    concavity_idx = 0
    
    if pos_cnt > neg_cnt:
        assert neg_cnt == 1
        concavity_idx = dirs.index(negatives[0])
    else:
        assert pos_cnt == 1
        concavity_idx = dirs.index(positives[0])
    
    return (concavity_idx + 3) % n

def project2D(points, face): #Projects the 3D points (mathutils.Vector) in the array onto the face returning the corresponding array of 2D points on that plane
    basis1 = (face.verts[0].co - face.verts[1].co).normalized()
    basis2 = face.normal.cross(basis1).normalized()
    origin = face.verts[0].co #TODO(fran): is it necessary to have an origin different from (0,0,0) for my needs?
    #x = (p - origin).dot(basis1)
    #y = (p - origin).dot(basis2)
    return [mathutils.Vector(((v - origin).dot(basis1), (v - origin).dot(basis2))) for v in points]

def project2DWithNormal(points, p1, p2, normal): #Projects the 3D points (mathutils.Vector) in the array onto the face (represented by one of the face's basis vectors (p2 - p1) and the face's normal) returning the corresponding array of 2D points on that plane
    basis1 = (p2 - p1).normalized()
    basis2 = normal.cross(basis1).normalized()
    origin = p1
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

def FixQuads(_do_cleanup = None, notifier = None):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    notify_err = lambda msg: (notifier and notifier({"ERROR"}, msg)) or print(msg)

    do_cleanup = True
    if _do_cleanup != None: do_cleanup = _do_cleanup
    visualization = not do_cleanup

    editmode()
    updatesel()

    mesh = bpy.context.active_object.data

    bm = bmesh.from_edit_mesh(mesh)

    if False: #testing
        import math
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        print(f"Treshold angle: {(pi/2)*.77}")
        #print([math.degrees(a) for a in get_face_angles(bm.faces[2732])])
        #print([abs(a-pi/2) for a in get_face_angles(bm.faces[2732])])
        
        #Manual 3D to 2D projection
        face = bm.faces[2732]
        
        concaveFace1 = [project2D([bm.verts[x].co for x in [3113,2368,2380,2362]], face)]
        concaveFace2 = [project2D([bm.verts[x].co for x in [3113,2362,2380,2368]], face)]
        
        convexFace1 = [project2D([bm.verts[x].co for x in [3113,2368,2367,2362]], face)]
        convexFace2 = [project2D([bm.verts[x].co for x in [3113,2362,2367,2368]], face)]
        
        for x in [concaveFace1, concaveFace2, convexFace1, convexFace2]: print(isConvex(x))

    assert bpy.context.tool_settings.mesh_select_mode[-1] == True, "Not in Face Selection Mode"

    faces = [f for f in bm.faces if f.select]

    assert all(isQuad(f) for f in faces), "Not all faces are quads"

    print(f"Quads : {[f.index for f in faces]}")

    assert all(len(link_faces(f)) <= 4 for f in faces), "Some quads have more than 4 faces around them (This shouldn't ever happen wtf)"

    #border_quads = [f for f in faces if len(set(link_faces(f)).intersection(set(faces))) == 1]

    #assert len(border_quads) <= 2, "There are more than 2 border quads"

    #connected_quads = []

    #print(f"Border quads : {[q.index for q in border_quads]}")

    verts_to_connect = []
    edges_to_dissolve = set()

    ''' #TODO(fran): see if this is a good replacement against all the geometry/shape checks im having to make to decide where to knife
    for f in faces: #verts to knife from/to will be the ones on the edgepairs with the biggest angle
        edgepairs = [[e for e in f.edges if v in e.verts] for v in f.verts]
        #print([[e.index for e in ep] for ep in edgepairs])
        assert all(len(ep) == 2 for ep in edgepairs), "I screwed something up in the edge pair calculation"
        
        vert_angles = [] #[(vert,angle),...]
        
        for ep in edgepairs:
            (shared_vert,) = set(ep[0].verts).intersection(set(ep[1].verts)) #Raises error if set doesnt contain only one element
            (v2,v3,) = set(ep[0].verts).symmetric_difference(set(ep[1].verts))
            vert_angles.append((shared_vert,(v2.co - shared_vert.co).angle(v3.co - shared_vert.co)))
        
        vert_angles.sort(reverse = True, key = lambda va: va[1])
        new_verts_to_connect = [v[0].index for v in vert_angles[0:2]]
        assert all(v not in verts_to_connect for v in new_verts_to_connect), "Error! We are trying to connect a vertex more than once"
        verts_to_connect.extend(new_verts_to_connect)

    assert len(verts_to_connect) % 2 == 0, "Somehow there's not an even number of vertices to connect/knife"
    '''

    final_face_strips = [] #array of arrays
    final_edge_strips = [] #array of arrays
    final_vert_strips = [] #array of arrays of tuples
    #_set_faces = set(faces)
    #border_faces = [f for f in faces if len(set(link_faces(f)).intersection(_set_faces)) <=1]
    border_faces = [f for f in bm.select_history] #thanks to the magic of the selection history I know for sure the borders (we will assume the user always selects borders, and that they do so making sure that no other tris are around that could confuse how that border continues)

    assert all(isinstance(f,bmesh.types.BMFace) for f in border_faces)

    strip_faces = set(f for f in faces if f not in border_faces)

    #for bf in border_faces: print(f"bf idx {bf.index}: {set(link_faces(bf)).intersection(_set_faces)}")

    print(f"Border Quads : {[x.index for x in border_faces]}")
    
    final_used_tris = dict() # {tri : set(border_face_who_used_it, ...), ...} Tris already taken by another strip are saved here so they cant be used again
    
    conflict_border_strips = [] #[(face_strips, edge_strips, vert_strips, used_tris_strips), ...]
    
    for bf in border_faces:
        face_strips = []
        edge_strips = []
        vert_strips = []
        
        used_tris_strips = [] # [[t_start, t_end], ...] tris that were used by each strip
        
        for lf in [x for x in link_faces(bf) if isTri(x)]:
            #Attempt to create a face strip
            face_strip = [bf]
            (shared_edge,) = set(lf.edges).intersection(bf.edges)
            edge_strip = [shared_edge]
            vert_strip = []
            used_tris_strip = [lf]
            
            #Decide how to cut the quads
            alt1_v0 = bf.verts[0]
            (alt1_v1,) = [v for v in bf.verts if not shares_edge_with(alt1_v0, v, bf)]
            alt1_verts = [alt1_v0, alt1_v1]
            alt2_verts = list(set(bf.verts).difference(set(alt1_verts)))
            assert len(alt2_verts) == 2
            
            CustomEdge = namedtuple('CustomEdge', 'verts')
            base_edges = [e for e in lf.edges if e != shared_edge]
            
            (alt1_quad_vert,) = [v for v in alt1_verts if v not in lf.verts]
            (alt1_3rd_edge,) = [e for e in [e for e in bf.edges if alt1_quad_vert in e.verts] if shared_edge in flatten([e.verts[0].link_edges, e.verts[1].link_edges])]
            alt1_edges = [alt1_3rd_edge, CustomEdge(alt1_verts)]
            
            (alt2_quad_vert,) = [v for v in alt2_verts if v not in lf.verts]
            (alt2_3rd_edge,) = [e for e in [e for e in bf.edges if alt2_quad_vert in e.verts] if shared_edge in flatten([e.verts[0].link_edges, e.verts[1].link_edges])]
            alt2_edges = [alt2_3rd_edge, CustomEdge(alt2_verts)]
            
            CustomFace = namedtuple('CustomFace', 'verts edges')
            
            alt1_face = CustomFace([*lf.verts, alt1_quad_vert], [*base_edges,*alt1_edges])
            alt2_face = CustomFace([*lf.verts, alt2_quad_vert], [*base_edges,*alt2_edges])
            
            alt_angle_difs = []
            alt_convexity = []
            
            for f in [alt1_face, alt2_face]: #verts to knife from/to will be the ones on the edgepairs with the biggest angle
                angle_difs = [abs(a-pi/2) for a in get_face_angles(f)] #Distance from angle to pi/2 (aka 90º, the perfect rectangle's angle), better angles will give smaller values here
                
                alt_angle_difs.append(max(angle_difs)) #save the alternative's worst angle
                
                #Manual 3D to 2D projection
                alt_ordered_points = project2D([v.co for v in get_ordered_vertices(f)],lf)
                
                alt_convexity.append(isConvex(alt_ordered_points))
            
            assert len(alt_angle_difs) == 2
            assert len(alt_convexity) == 2
            
            go_alt1 = lambda: vert_strip.append(alt1_verts) #Alternative 1 wins
            go_alt2 = lambda: vert_strip.append(alt2_verts) #Alternative 2 wins
            
            if alt_angle_difs[0] == alt_angle_difs[1]:
                if alt_convexity[0] == alt_convexity[1]: continue #fail: impossible to decide
                elif alt_convexity[0]: go_alt1()
                else: go_alt2()
            elif alt_angle_difs[0] < alt_angle_difs[1]:
                if alt_convexity[0]: go_alt1()
                elif alt_convexity[1]: go_alt2()
                else: continue #fail: impossible to decide
            else:
                if alt_convexity[1]: go_alt2()
                elif alt_convexity[0]: go_alt1()
                else: continue #fail: impossible to decide
            
            
            fail = False
            cnt = 0
            while True:
                cnt += 1
                assert cnt < 20
                (follow_edge, ) = (e for e in face_strip[-1].edges if all([x not in shared_edge.verts for x in e.verts]))
                edge_strip.append(follow_edge)
                elf = [x for x in follow_edge.link_faces if x != face_strip[-1]]
                shared_edge = follow_edge
                
                if len(elf) == 1:
                    elf = elf[0]
                    if isTri(elf):
                        #TODO(fran): if the gen face with the try is not good, then  we should start undoing the previous knifings and try other combinations, that could get quite tricky, so for now we simply error and do not convert anything, TODO: at least try to undo just the last connection
                        used_tris_strip.append(elf)
                        
                        (quad_v,) = [v for v in vert_strip[-1] if v not in elf.verts]
                        inbetween_vs = [v for v in face_strip[-1].verts if v in elf.verts] #TODO(fran): there may be better/safer ways to do this without looking into face_strip
                        (last_tri_v,) = [v for v in elf.verts if v not in inbetween_vs]
                        assert len(inbetween_vs) == 2
                        
                        gen_quad_ordered_verts = [inbetween_vs[0], quad_v, inbetween_vs[1], last_tri_v]
                        gen_quad_ordered_points = [x.co for x in gen_quad_ordered_verts]
                        
                        corrected_vert_strip_section = [v for v in face_strip[-1].verts if v not in vert_strip[-1]]
                        assert len(corrected_vert_strip_section) == 2
                        
                        gen_quad_convexity = isConvex(project2DWithNormal(gen_quad_ordered_points, *gen_quad_ordered_points[:2], elf.normal)) #TODO(fran): here I should calculate and use the newly generated quad's normal
                        
                        if not gen_quad_convexity:
                            print(f"WARNING: Shape would not be convex at Face of index {elf.index} (this probably indicates a prior analysis mistake when choosing the connection points): vertex knifing for the previous face (of index {face_strip[-1].index}) will be flipped from what it was originally decided")
                            #TODO(fran): analyse the new convexity for both this tri and the previous quad's generated quads to make sure that we need to flip this decision and not an older one
                            vert_strip[-1] = corrected_vert_strip_section #TODO(fran): before making the flip I should at least make sure the convexity of this alternative is good, otherwise I should simply fail
                            #fail = True
                        
                        break #face strip is complete
                    elif elf in strip_faces:
                        face_strip.append(elf) #continue making the face strip
                        (strip_v1,) = [v for v in follow_edge.verts if v not in vert_strip[-1]]
                        (strip_v2_follow,) = [v for v in follow_edge.verts if v != strip_v1]
                        (strip_v2,) = [v for v in elf.verts if shares_edge_with(v, strip_v2_follow, elf) and v not in follow_edge.verts]
                        
                        (already_connected_v,) = [v for v in vert_strip[-1] if shares_edge_with(v, strip_v2, elf)]
                        (other_v,) = [v for v in vert_strip[-1] if v != already_connected_v]
                        generated_quad_ordered_verts = [already_connected_v, strip_v2, strip_v1, other_v]
                        generated_quad_ordered_points = [x.co for x in generated_quad_ordered_verts]
                        
                        corrected_vert_strip_section = [v for v in elf.verts if v not in [strip_v1, strip_v2]]
                        assert len(corrected_vert_strip_section) == 2
                        
                        generated_quad_convexity = isConvex(project2DWithNormal(generated_quad_ordered_points, *generated_quad_ordered_points[:2], elf.normal)) #TODO(fran): here I should calculate and use the newly generated quad's normal
                        
                        elf_ordered_verts = get_ordered_vertices(elf)
                        elf_ordered_points_projected = project2D([x.co for x in elf_ordered_verts], elf)
                        
                        #TODO(fran): we know the dissolve edges are correct, thus we could check if disolving would cause a vertex to dissapear, in which case we must make the knifing along that vertex for it to continue existing, since we know we should NEVER have to add or remove vertices (this is made a little more complex due to the fact that whether the vertex gets dissolved sometimes depends not only on the current knifing but also the next one, we need to implement future analysis so that we can move forwards and see how the strip would progress)
                        
                        #TODO(fran): check for known good shapes (quad, diamond, etc), or maybe safer, check for known bad shapes (pyramid, etc). For this we'd need to look at the gradients between points, or at very specific configurations and orders of angles
                        
                        #TODO(fran): the other option would be to go back to comparing angles between alternatives and taking the one that has the best overall angles
                                                
                        if not isConvex(elf_ordered_points_projected):
                            concavity_vert = elf_ordered_verts[findConcavityVertexIdx(elf_ordered_points_projected)]
                            print(f"WARNING: Face {elf.index} was not convex (concavity vertex index: {concavity_vert.index}), we are forced to knife it along the concavity point in order to not generate a quad that spills over the original one's bounds")
                            vert_strip.append(corrected_vert_strip_section if concavity_vert in corrected_vert_strip_section else [strip_v1, strip_v2])
                        elif all(abs(a-pi/2) > (pi/2)*.77 for a in get_face_angles(elf)): #HACK: the better way to do this would be to find out if the face crosses two different face loops
                            #Face has a stretched diamond shape, probably going across two faces loops, it requires the opposite vertex knifing
                            print(f"WARNING: DIAMOND SHAPE FOUND at Face of index {elf.index}: vertex knifing for this face will be done opposite the others")
                            #TODO(fran): I should actually check again that the newly generated face would have a rectangular shape, otherwise I shouldnt change the verts
                            vert_strip.append(corrected_vert_strip_section)
                            #TODO(fran): we dont check if the first face was a diamond shape, in which case we should do the same
                        elif not generated_quad_convexity:
                            print(f"WARNING: Shape would not be convex at Face of index {elf.index} (this probably indicates a change in face direction): vertex knifing for this face will be done opposite the others")
                            vert_strip.append(corrected_vert_strip_section)
                        elif any(abs(a-pi/2) > (pi/2)*.864 for a in get_face_angles_from_points(generated_quad_ordered_points)):
                            #Reminder: Previous thresholds: (pi/2)*.8717
                            print(f"WARNING: Shape would have a very bad angle at Face of index {elf.index} with candidate vertices {[x.index for x in generated_quad_ordered_verts]} (this probably indicates a change in face direction): vertex knifing for this face will be done opposite the others")
                            #TODO(fran): im thinking of adding a check for a huge angle and a small one, similar to this one's condition but checking for a big and a small angle)
                            vert_strip.append(corrected_vert_strip_section)
                        else:
                            vert_strip.append([strip_v1, strip_v2])
                    else:
                        fail = True
                        break
                else:
                    fail = True
                    break
            if not fail:
                face_strips.append(face_strip)
                edge_strips.append(edge_strip)
                vert_strips.append(vert_strip)
                used_tris_strips.append(used_tris_strip)
                
        
        if len(face_strips) > 1:
            #We have a conflict, there's more than one possible solution, here we will remove duplicated solutions, if more than one alternatie remains after that then this face strips will go to the next stage of analysis after all other border faces have been analysed
            assert all(len(x) == 2 for x in used_tris_strips)
            assert len(face_strips) <= 4
            
            #First we remove duplicated solutions
            for i, fs in reversed(list(enumerate(face_strips))):
                for j, fs2 in enumerate(face_strips):
                    if i != j and all(x in used_tris_strips[j] for x in used_tris_strips[i]): #TODO(fran): we should actually also check that their edge and vert strips are the same, but that is a whole other job for later
                        assert len(fs) == len(fs2) == 1 #Duplicate cases should only happen for single face strips
                        if fs[0] == fs2[0]:
                            del face_strips[i]
                            del edge_strips[i]
                            del vert_strips[i]
                            del used_tris_strips[i]
                            break
                
        
        if len(face_strips) == 1:
            final_face_strips.extend(face_strips)
            final_edge_strips.extend(edge_strips)
            final_vert_strips.extend(vert_strips)
            for t in flatten(used_tris_strips): #NOTE: for now we simply flatten the entire list, later when we accept multiple face_strips we should input them separately into the dict to know which alternative should be ignored and which can still be valid
                final_used_tris.setdefault(t, set()).add(bf)
        elif len(face_strips) == 0:
            print(f"WARNING: Border face of index {bf.index} has no possible strips to be fixed with, did you make a wrong selection?")
        else:
            conflict_border_strips.append((face_strips, edge_strips, vert_strips, used_tris_strips))
            print(f"WARNING: Border face of index {bf.index} has more than one possible strip that it can generate, it will be anaysed after all the others to try and resolve which alternative is best")
    
    for c in conflict_border_strips:
        (face_strips, edge_strips, vert_strips, used_tris_strips,) = c
        
        #Look for already used tris from known good strips and reject those alternatives for this strip
        for i, fs in reversed(list(enumerate(face_strips))):
            if any(t in final_used_tris for t in used_tris_strips[i]):
                del face_strips[i]
                del edge_strips[i]
                del vert_strips[i]
                del used_tris_strips[i]
                break
        
        #TODO(fran): add more analysis algorithms
        
        if len(face_strips) == 1:
            final_face_strips.extend(face_strips)
            final_edge_strips.extend(edge_strips)
            final_vert_strips.extend(vert_strips)
            for t in flatten(used_tris_strips):
                final_used_tris.setdefault(t, set()).add(bf)
        elif len(face_strips) == 0:
            print(f"WARNING: Border face of index {bf.index} has no possible strips to be fixed with, did you make a wrong selection? Or maybe this indicates an ERROR in the analysis") #TODO(fran): find out if this can come from analysis errors
        else:
            notify_err(f"ERROR: Border face of index {bf.index} has multiple valid strip alternatives and the analyser cant make a choice, unfortunately we will skip fixing it. (Selecting more border faces that surround this one may give the analyser the context it needs to make a decision)")
            
    
    
    border_faces_idx = [x.index for x in border_faces]
    final_face_strips_idx = [[s.index for s in x] for x in final_face_strips]
    final_edge_strips_idx = [[s.index for s in x] for x in final_edge_strips]
    final_vert_strips_idx = [[[s[0].index, s[1].index] for s in x] for x in final_vert_strips]
    
    if visualization:
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT')
        #bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='EDGE')
        #bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='FACE')
        objectmode()
        '''
        for v in verts_to_connect:
            #mesh.edges[ff[1]].select = True
            mesh.vertices[v].select = True 
        '''
        #for f in border_faces_idx: mesh.polygons[f].select = True 
        
        #for fs in final_face_strips_idx: 
        #    for f in fs: mesh.polygons[f].select = True
        
        #for es in final_edge_strips_idx: 
        #    for e in es: mesh.edges[e].select = True 
        
        for vs in final_vert_strips_idx: 
            for v in vs: mesh.vertices[v[0]].select, mesh.vertices[v[1]].select = True, True

    elif do_cleanup:
        editmode()
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT')
        for vs in final_vert_strips_idx: 
            #TODO(fran): all the context changes make this loop very slow. Optimizations: connect multiple tris at a time (make sure they are not connected/near other selected tris), or directly operate over the bmesh (with bmesh.ops and bmesh.utils) and then update the real mesh
            #NOTE: the fail case occurs when the selected vertices make a face be fully selected (that's when some vertices dont get connected/paired), we should make groups of non face making vertices and select them all at once so this loop gets a lot faster
            #Another option would be to select everything, connect it, then go back and test that a new edge was made between each vertex pair, if it was we remove it from the list, otherwise we keep it and run the loop again
            for v in vs:
                objectmode()
                mesh.vertices[v[0]].select, mesh.vertices[v[1]].select = True, True
                editmode()
                updatesel()
                bpy.ops.mesh.vert_connect()
                bpy.ops.mesh.select_all(action='DESELECT')
        
        editmode()
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='EDGE')
        objectmode()
        for es in final_edge_strips_idx:
            for e in es: mesh.edges[e].select = True
        editmode()
        updatesel()
        bpy.ops.mesh.dissolve_edges()
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='FACE')

    editmode()
    updatesel()

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("")
    bm.free()

if not registering:
    FixQuads(False)
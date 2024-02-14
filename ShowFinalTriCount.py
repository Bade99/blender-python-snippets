#Prints to the console a table with vertex, edge, face and tri information for all the objects currently selected
import bpy

def print(data):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CONSOLE':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT")

sorted = True
if not sorted:
    verts, edges, faces, tris_aprox = 0, 0, 0, 0
    dg = bpy.context.evaluated_depsgraph_get()  # Getting the dependency graph
    print('-'*50)
    for obj in bpy.context.selected_objects:

        obj = obj.evaluated_get(dg)
        # This gives the evaluated version of the object. Aka with all modifiers and deformations applied.  
        mesh = obj.to_mesh()  # Turn it into the mesh data block we want
        v_cnt, e_cnt, f_cnt = len(mesh.vertices), len(mesh.edges), len(mesh.polygons)
        t_cnt_aprox = sum((len(f.vertices) - 2) for f in mesh.polygons)
        print(f' Object : {obj.name}')
        print(f'  Vertices : {v_cnt} | Edges : {e_cnt} | Faces : {f_cnt} | Tris (aprox) : {t_cnt_aprox}')
        verts += v_cnt
        edges += e_cnt
        faces += f_cnt
        tris_aprox += t_cnt_aprox
    print(f'Total : Verts : {verts} | Edges : {edges} | Faces : {faces} | Tris (aprox) : {tris_aprox}')
    print('-'*50)
else:
    tot_v, tot_e, tot_f, tot_t = 0,0,0,0
    objs = [] #[(name, verts, edges, faces, tris),...]
    dg = bpy.context.evaluated_depsgraph_get()
    print('-'*100)
    for obj in bpy.context.selected_objects:
        name, verts, edges, faces, tris_aprox = '', 0, 0, 0, 0
        obj = obj.evaluated_get(dg)
        mesh = obj.to_mesh()
        v_cnt, e_cnt, f_cnt = len(mesh.vertices), len(mesh.edges), len(mesh.polygons)
        t_cnt = sum((len(f.vertices) - 2) for f in mesh.polygons)
        objs.append((obj.name, v_cnt, e_cnt, f_cnt, t_cnt))
        tot_v += v_cnt
        tot_e += e_cnt
        tot_f += f_cnt
        tot_t += t_cnt
    objs.sort(key = lambda o: o[4]) # sort by tri count
    
    name_len = len(max(objs, key = lambda o: len(o[0]))[0]) + 1
    
    template_header = "|{0:" + str(name_len+1) + "}|{1:10}  |{2:10}  |{3:10}  |{4:10}  |"
    template = "| {0:" + str(name_len) + "}|{1:10} v|{2:10} e|{3:10} f|{4:10} t|" # Number after the ':' is the column width
    
    print(template_header.format("OBJECT", "VERTS", "EDGES", "FACES", "TRIS"))
    for o in objs: print(template.format(*o))
    print("")
    print(f'Total : Verts : {tot_v} | Edges : {tot_e} | Faces : {tot_f} | Tris : {tot_t}')
    print('-'*100)
    print("")
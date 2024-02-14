#Constructs separate individually addressable merged objects from a group of selected objects usually containing array & mirror modifiers. It deduces the number of objects to create based on the proximity of the subobjects to each other, joining them or otherwise generating separate objects.
import bpy

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

make_single_object = True

objectmode()
wheel_obj_name = bpy.context.active_object.name
objs_to_process = bpy.context.selected_objects.copy()
for o in objs_to_process:
    bpy.context.view_layer.objects.active = o
    #bpy.ops.object.make_single_user(object=False, obdata=True, material=True, animation=False, obdata_animation=False)
    #TODO(fran): here is a pretty annoying problem, when working with materials with Link set to Object instead of the default Data blender goes back to the material that was in Data when using Separate by Loose Parts. Annoying solution: we'd need to check if a material is set to Object, save it, apply all modifiers & separate by loose parts, then reapply the material and in the same material slot.
    first_time = True
    for m in o.modifiers:
        if m.type != 'EDGE_SPLIT' and m.show_viewport:
            if first_time:
                first_time = False
                bpy.ops.object.modifier_apply(modifier=m.name, single_user=True)
            else:
                bpy.ops.object.modifier_apply(modifier=m.name)


wheel_obj = bpy.data.objects[wheel_obj_name]

wheel_collection = wheel_obj.users_collection[0]

objs_to_process.remove(wheel_obj)

bpy.ops.object.select_all(action='DESELECT')
bpy.context.view_layer.objects.active = wheel_obj
wheel_obj.select_set(True)
editmode()
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.separate(type='LOOSE') #BIG TODO(fran): separating by loose parts is terrible because I usually have loose parts even in the single original mesh, I need to use something else like DupliVerts (with instancing on faces). OR I could merge every object inside the wheel empty into a single object (which would solve the 'too many' submeshes problem).
objectmode()
assert len(bpy.context.selected_objects) >= 1
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

objs_selected = bpy.context.selected_objects.copy()
wheel_cnt = len(objs_selected)
bpy.ops.object.select_all(action='DESELECT')

wheel_container_names = []

for i, o in enumerate(objs_selected):
    bpy.ops.object.empty_add(type='PLAIN_AXES', radius=0.1, align='WORLD', location=o.location, scale=(1, 1, 1))
    empty = bpy.context.active_object
    empty.name = wheel_obj_name + "_wheel_container_" + str(i)
    wheel_collection.objects.link(empty)
    empty.users_collection[1].objects.unlink(empty) #HACK: I should actually check which collections it is in, and if it's different from the wheel_collection then remove the object from there
    #TODO(fran): place the object on the same level/container in the object viewer on the right
    wheel_container_names.append(empty.name)
    o.select_set(True)
    assert len(bpy.context.selected_objects) == 2
    bpy.ops.object.parent_set(type='OBJECT')

wheel_containers = [bpy.data.objects[x] for x in wheel_container_names]

bpy.context.view_layer.objects.active = None

for o in objs_to_process:
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    editmode()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.separate(type='LOOSE')
    objectmode()
    objs_selected = bpy.context.selected_objects.copy()
    assert len(objs_selected) != 0 and len(objs_selected) % wheel_cnt == 0, f"An invalid number of subobjects was generated for {o.name}"
    subobjects_per_wheel = len(objs_selected) / wheel_cnt
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    bpy.ops.object.select_all(action='DESELECT')
    
    containers = [[x, subobjects_per_wheel] for x in wheel_containers]
    for o in objs_selected:
        bpy.ops.object.select_all(action='DESELECT')
        cont = min(containers, key= lambda x: (x[0].location - o.location).length_squared)
        cont[1] -= 1
        assert cont[1] >= 0
        
        container = cont[0]
        
        bpy.context.scene.cursor.location = container.location
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
                
        container.select_set(True)
        bpy.context.view_layer.objects.active = container
        assert len(bpy.context.selected_objects) == 2
        bpy.ops.object.parent_set(type='OBJECT')
    
if make_single_object:
    for wheel in wheel_containers:
        bpy.ops.object.select_all(action='DESELECT')
        assert len(wheel.children) > 0
        for c in wheel.children_recursive:
            c.select_set(True)
        bpy.context.view_layer.objects.active = wheel.children[0]
        bpy.ops.object.join()

from TGMMlibraries import lineageTree
import numpy as np
import os
import ast
from mamut_xml_templates import *
import sys

def read_param_file():
    if (sys.argv[1] is not None) and (os.path.exists(sys.argv[1])):
        f_names = [sys.argv[1]]
        #print f_names + "\n"
    else:
        p_param = input('Please enter the path to the parameter file/folder:\n')
        p_param = p_param.replace('"', '')
        p_param = p_param.replace("'", '')
        p_param = p_param.replace(" ", '')
        if p_param[-4:] == '.txt':
            f_names = [p_param]
        else:
            f_names = [os.path.join(p_param, f) for f in os.listdir(p_param) if '.txt' in f and not '~' in f]

    for file_name in f_names:
        f = open(file_name)
        lines = f.readlines()
        f.close()
        param_dict = {}

        for line in lines:
            # Strip whitespace and ignore empty lines or lines starting with '#'
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split the line at '=' and strip whitespace
            parts = line.split('=', 1)
            if len(parts) != 2:
                print(f"Warning: Line '{line}' is not in the expected format.")
                continue

            param_name, param_value = parts
            param_name = param_name.strip()
            param_value = param_value.split('#')[0].strip()  # Remove comment

            #if '[' in param_value and ']' in param_value: #array
            if param_name in ['tissues_pixel_values', 'downsampling', 'tissue_ids']:
                # Safely evaluate the string as a Python list
                python_list = ast.literal_eval(param_value)

                # Determine if the list contains floats or integers
                is_float = any(isinstance(item, float) for item in python_list)

                # Convert the list to a NumPy array with the appropriate data type
                dtype = float if is_float else int
                param_dict[param_name] = np.array(python_list, dtype=dtype)

            elif param_name in ['label_names','tissue_names']:
                # Safely evaluate the string as a list
                string_list = ast.literal_eval(param_value)

                # Convert th list to a NumPy array of strings
                param_dict[param_name] = np.array(string_list, dtype=str)
            elif 'time' in param_name:
                if param_value.isdigit():
                    param_dict[param_name] = int(param_value)
                else:
                    param_dict[param_name] = float(param_value)
            else:
                param_dict[param_name] = param_value
            #print( param_name + '=' + param_value )

#        i = 1
#        nb_lines = len(lines)
#        delimeter = lines[0]
#        delimeter = delimeter.rstrip()
#        while i < nb_lines:
#            l = lines[i]
#            split_line = l.split(delimeter)
#            param_name = split_line[0]
#            if param_name in ['labels', 'downsampling', 'tissue_ids', 'tissue_names']:
#                name = param_name
#                out = []
#                while (name == param_name or param_name == '') and  i < nb_lines:
#                    if split_line[1].strip().isdigit():
#                        out += [int(split_line[1])]
#                    else:
#                        out += [(split_line[1].strip())]
#                    i += 1
#                    if i < nb_lines:
#                        l = lines[i]
#                        split_line = l.split(delimeter)
#                        param_name = split_line[0]
#                param_dict[name] = np.array(out)
#            elif param_name in ['label_names']:
#                name = param_name
#                out = []
#                while (name == param_name or param_name == '') and  i < nb_lines:
#                    out += [split_line[1].replace('\n', '').replace('\r', '')]
#                    i += 1
#                    if i < nb_lines:
#                        l = lines[i]
#                        split_line = l.split(delimeter)
#                        param_name = split_line[0]
#                param_dict[name] = np.array(out)
#            else:
#                param_dict[param_name] = split_line[1].strip()
#                i += 1
#            if param_name == 'time':
#                param_dict[param_name] = int(split_line[1])
        path_SVF = param_dict.get('path_to_SVF', '.')
        path_DB = param_dict.get('path_to_DB', '')
        path_output = param_dict.get('path_output', '.')
        path_LUT= param_dict.get('path_lut', '.')
        #tissue_ids = param_dict.get('tissue_ids', [])
        tissue_names = param_dict.get('tissue_names', [])
        begin = int(param_dict.get('begin', None))
        end = int(param_dict.get('end', None))
        do_mercator = bool(int(param_dict.get('do_mercator', '0')))
        filename = param_dict.get('filename', '.')
        folder = param_dict.get('folder', '.')
        v_size = float(param_dict.get('v_size', 0.))
        dT = float(param_dict.get('dT', 1.))

    return (path_SVF, path_DB, path_output, path_LUT,
        tissue_names, begin, end, filename, folder, v_size, dT, do_mercator)

def main():
    try:
        (path_SVF, path_to_DB, path_output, path_LUT,
            tissue_names, begin, end, filename, folder, v_size, dT, do_mercator) = read_param_file()
    except Exception as e:
        print("Failed at reading the configuration file.")
        print("Error: %s"%e)
        raise e
    
    SVF = lineageTree(path_SVF)

    # f = open(path_to_DB)
    # lines = f.readline()
    # f.close()
    #print "Do mercator %s"%do_mercator
    if os.path.exists(path_to_DB):
        DATA = np.loadtxt(path_to_DB, delimiter = ',', skiprows = 1, usecols = (0, 6,7, 9, 10))
        tracking_value = dict(DATA[:, (0, 3)])
        sphere_coord_theta = dict(DATA[:, (0, 1)])
        sphere_coord_phi = dict(DATA[:, (0, 2)])
        kept_nodes = [c for c in SVF.nodes if tracking_value[c] >= 0]
        kept_nodes_set = set(kept_nodes)
        #t_id_2_N = dict(list(zip(tissue_ids, tissue_names)))
    else:
        kept_nodes = SVF.nodes
        kept_nodes_set = set(kept_nodes)
        tracking_value = {c:1 for c in kept_nodes}


    kept_times = list(range(begin, end+1))
    first_nodes = [c for c in SVF.time_nodes[min(kept_times)] if c in kept_nodes_set]

    if not os.path.exists(os.path.dirname(path_output)):
        os.makedirs(os.path.dirname(path_output))
    with open(path_output, 'w') as output:

        output.write(begin_template)

        # Begin AllSpots.
        output.write(allspots_template.format(nspots=len(kept_nodes)))
        
        # Loop through lists of spots to try to center the model
        Q = []
        abs_center = [ 0.0,0.0,0.0 ]
        if do_mercator:
            for t in kept_times:
               for c in SVF.time_nodes[t]:
                   SVF.pos[c][0] = 20*np.arctan(np.exp(sphere_coord_theta[c]))-(np.pi/2 )
                   SVF.pos[c][1] = 10*sphere_coord_phi[c]
                   SVF.pos[c][2] = 0.0
                   Q += [SVF.pos[c]]
            abs_center = np.median(Q,axis=0)  
        elif v_size > 0:
            for t in kept_times:
               for c in SVF.time_nodes[t]:
                   SVF.pos[c][0] *= v_size
                   SVF.pos[c][1] *= v_size
                   SVF.pos[c][2] *= v_size
        else:
            for t in kept_times:
               for c in SVF.time_nodes[t]:
                   Q += [SVF.pos[c]]
            abs_center = np.median(Q,axis=0) 
            
        # Loop through lists of spots.
        for t in kept_times:
            cells = kept_nodes_set.intersection(SVF.time_nodes.get(t, []))
            if cells != []:
                output.write(inframe_template.format(frame=t))
                for c in cells:
                    #c_value = tracking_value[c]  # Use get() to handle cases where c is not in tracking_value

                    #print(f"C: {c} {c_value}")
                    output.write(spot_template.format(id=c, name=c, frame=t, t_id=tracking_value[c],
                                                      x=SVF.pos[c][0]-abs_center[0],
                                                      y=SVF.pos[c][1]-abs_center[1],
                                                      z=SVF.pos[c][2]-abs_center[2],
                                                      t_name=tissue_names[int(tracking_value[c])], #t_id_2_N.get(tracking_value[c], '?')
                                                      t_color=-65535
                                                      ))
                output.write(inframe_end_template)
            else:
                output.write(inframe_empty_template.format(frame=t))


        # End AllSpots.
        output.write(allspots_end_template)

        all_tracks = []
        roots = set(kept_nodes).difference(SVF.predecessor).union(first_nodes)
        last_time = max(kept_times)
        for c in roots:
            track = [c]
            while track[-1] in SVF.successor and SVF.time[track[-1]]<last_time:
                track += SVF.successor[track[-1]]
            all_tracks += [track]

        # Begin AllTracks.
        output.write(alltracks_template)

        for track_id, track in enumerate(all_tracks[::-1]):
            stop = SVF.time[track[-1]]
            duration = len(track)
            output.write(track_template.format(id=track_id+1, duration=duration, 
                                               start=SVF.time[track[0]], stop=stop, nspots=len(track),
                                               displacement=np.linalg.norm(SVF.pos[track[0]]-SVF.pos[track[-1]])))
            for c in track[:-1]:
                displacement = np.linalg.norm(SVF.pos[c] - SVF.pos[SVF.successor[c][0]]) * v_size
                velocity = displacement / dT
                output.write(edge_template.format(source_id=c, target_id=SVF.successor[c][0],
                                                  # t_name=t_id_2_N.get(tracking_value[c], '?'),
                                                  t_name=tissue_names[int(tracking_value[c])], #t_id_2_N.get(tracking_value[c], '?')
                                                  t_color=-65535,
                                                  velocity=velocity, displacement=displacement,
                                                  t_id=tracking_value[c], time=SVF.time[c]))
            output.write(track_end_template)

        # End AllTracks.
        output.write(alltracks_end_template)

        # Filtered tracks.
        output.write(filteredtracks_start_template)
        for track_id, track in enumerate(all_tracks[::-1]):
            output.write(filteredtracks_template.format(t_id=track_id+1))
        output.write(filteredtracks_end_template)

        # End XML file.
        output.write(end_template.format(image_data=im_data_template.format(filename=filename, folder=folder)))

if __name__ == '__main__':
    main()

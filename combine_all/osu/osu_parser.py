import math
import os
import random

import EEG_DL.Utils.config as config

def parse_metadata(filepath):
    meta = {'Title': 'Unknown', 'Artist': 'Unknown', 'AudioFilename': None, 'Version': '', 'filepath': filepath, 'difficulty': 0}
    if not os.path.exists(filepath): return meta
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        in_hit = False
        count = 0
        for line in lines:
            line = line.strip()
            if not line: continue
            if line.startswith('Title:'): meta['Title'] = line.split(':', 1)[1].strip()
            if line.startswith('Artist:'): meta['Artist'] = line.split(':', 1)[1].strip()
            if line.startswith('Version:'): meta['Version'] = line.split(':', 1)[1].strip()
            if line.startswith('AudioFilename:'): meta['AudioFilename'] = line.split(':', 1)[1].strip()
            if line.startswith('[HitObjects]'):
                in_hit = True
                continue
            if in_hit and ',' in line:
                count += 1
        meta['difficulty'] = count
    except: pass
    return meta

def parse_osu_map(filepath):
    meta = parse_metadata(filepath)
    hit_objects = []
    timing_points = [] 
    key_count = 4 
    section = None
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            if line.startswith('['):
                section = line
                continue
            
            if section == '[Difficulty]' and line.startswith('CircleSize:'):
                key_count = int(float(line.split(':')[1]))
            
            if section == '[TimingPoints]':
                parts = line.split(',')
                if len(parts) < 2: continue
                try:
                    t = float(parts[0])
                    beat_len = float(parts[1])
                    uninherited = int(parts[6]) if len(parts) > 6 else 1
                    
                    if uninherited == 1 and beat_len > 0:
                        timing_points.append({'time': t, 'beat_length': beat_len})
                except: pass

            if section == '[HitObjects]':
                parts = line.split(',')
                if len(parts) < 4: continue
                x = int(parts[0])
                time_ms = int(parts[2])
                obj_type = int(parts[3])
                
                lane = 0 
                
                if (obj_type & 128):
                    extras = parts[5].split(':')
                    hit_objects.append({'lane': lane, 'time': time_ms, 'type': 'hold', 'end': int(extras[0]), 'hit': False})
                else:
                    hit_objects.append({'lane': lane, 'time': time_ms, 'type': 'tap', 'end': 0, 'hit': False})
        
        hit_objects.sort(key=lambda x: x['time'])
        timing_points.sort(key=lambda x: x['time'])
        
        filtered_notes = []
        
        #間隔為4s
        TARGET_MIN_MS = 4000
        SNAP_ERROR_THRESHOLD = 0.08 
        
        INITIAL_BLANK_MS = 3000
        
        next_target_time = -999999 
        tp_index = 0
        
        for n in hit_objects:
            t = n['time']
            
            # 過濾前三秒note
            if t < INITIAL_BLANK_MS:
                continue
            
            while tp_index < len(timing_points) - 1 and timing_points[tp_index+1]['time'] <= t:
                tp_index += 1
            
            if not timing_points:
                filtered_notes.append(n)
                continue

            current_tp = timing_points[tp_index]
            beat_len = current_tp['beat_length'] 
            start_time = current_tp['time']
            
            if t < next_target_time - 50:
                continue
            
            local_beat = (t - start_time) / beat_len
            nearest_whole_beat = round(local_beat)
            
            if abs(local_beat - nearest_whole_beat) <= SNAP_ERROR_THRESHOLD:
                
                if len(filtered_notes) > 0 and abs(filtered_notes[-1]['time'] - t) < 10:
                    continue
                    
                filtered_notes.append(n)
                
                raw_beats_needed = TARGET_MIN_MS / beat_len
                required_beats = 1
                while required_beats < raw_beats_needed:
                    required_beats *= 2  
                
                next_target_time = t + (required_beats * beat_len)


        # 平衡note分佈到 N_Class 個 lane
        numClasses = config.N_Class
        total_notes = len(filtered_notes)
        print(f"[DEBUG] Total notes: {total_notes}, N_Class: {numClasses}")
        
        if total_notes > 0:
            base_count = total_notes // numClasses
            remainder = total_notes % numClasses
            
            balanced_lanes = []
            for classIdx in range(numClasses):
                balanced_lanes.extend([classIdx] * base_count)
            
            if remainder > 0:
                extra_lanes = random.sample(range(numClasses), remainder)
                balanced_lanes.extend(extra_lanes)
            
            random.shuffle(balanced_lanes)
            
            for i in range(total_notes):
                filtered_notes[i]['lane'] = balanced_lanes[i]

        hit_objects = filtered_notes

    except Exception as e:
        print(f"fail {filepath}: {e}")
        
    return {'meta': meta, 'notes': hit_objects}
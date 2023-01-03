import re
import json
import jsonlines
import os
import glob
import matplotlib.pyplot as plt
import shutil

from transformers import AutoTokenizer

character = re.compile(r'([A-Z ]+\n)')
enter = re.compile(r'(Enter [\w .?\-\",;]+\n)')
exeunt = re.compile(r'((Exit|Exeunt)[\w .?\-\'\",;]*\n)')
act_scene = re.compile(r'(SCENE[\w .?\"\'\-,;]*\n|ACT[\w .?\"\'\-,;]*\n)')


def get_scenes(filename):# divide by scene to get local dialogues
    with open(filename, 'r') as f:
        text = f.read()
    scenes = re.split(act_scene, text)
    return scenes


def delete_scene_headings(scenes):
    for i, scene in enumerate(scenes):
        if re.match(act_scene, scene):
            scenes.pop(i)
        elif scene == '':
            scenes.pop(i)
    return scenes


def delete_entrance_exits(scene):
    scene_stripped = re.sub(enter, '', scene)
    scene_stripped = re.sub(exeunt, '', scene_stripped)
    return scene_stripped


def get_dialogues(scene):
    # split scenes into dialogue, with character_name + line
    dialogues = []
    lines_with_characters = re.split(character, scene)
    for i, line in enumerate(lines_with_characters):
        if line == '':
            continue
        if re.match(character, line):
            dialogues.append({'character_and_line': line + lines_with_characters[i+1]})
    return dialogues


def group_dialogues(dialogues, tokenizer, taskname='eme-seq2seq', max_input_length=1024):
    graduated_blocks = []
    for i, d in enumerate(dialogues):
        if i == 0:
            context = d['character_and_line']
            graduated_blocks.append({'taskname': taskname, 'context': context, 'response': dialogues[i+1]['character_and_line'], 'context_len': len(tokenizer(context)['input_ids'])})
        elif i == 1:
            continue
        else:
            context = ' '.join([k['character_and_line'] for k in dialogues[0:i]])
            # check for context length
            tokenized = tokenizer(context)['input_ids']
            len_tokens = len(tokenized)
            print('Length of context:', len_tokens)
            if len_tokens > max_input_length:
                print('long context! attempting to truncate from the beginning')
                # set index for stepping through
                ind = 0
                while len_tokens > max_input_length:
                    print('inside while loop; truncating')
                    # truncate from the beginning
                    context = ' '.join([k['character_and_line'] for k in dialogues[ind+1:i]])
                    tokenized = tokenizer(context)['input_ids']
                    len_tokens = len(tokenized)
                    ind += 1

            new_group = {'taskname': taskname, 'context': context, 'response': d['character_and_line'], 'context_len': len_tokens}
            graduated_blocks.append(new_group)
    return graduated_blocks


def save_text_as_dialogues(textfile, output_dir, tokenizer_name='gpt2'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    scenes = get_scenes(textfile)
    scenes = delete_scene_headings(scenes)
    for i, scene in enumerate(scenes):
        print(f'DEBUG: working on scene {i} of {len(scenes)}')
        if scene == '':
            print('WARNING: found a blank scene')
            continue
        scene = delete_entrance_exits(scene)
        dialogues = get_dialogues(scene)
        with jsonlines.open(os.path.join(output_dir, 'dialogues_'+str(i)+'.json'), 'w') as writer:
            writer.write_all(dialogues)


def preprocess_to_groups(play_dialogue_dir, output_dir, tokenizer, max_input_length=1024):
    scene_files = glob.glob(play_dialogue_dir+'/*')
    for i, scene_f in enumerate(scene_files):
        dialogues = read_json_lines(scene_f)
        if len(dialogues) > 1:  # some scenes are monologues :-P
            dialogues = group_dialogues(dialogues, tokenizer, max_input_length=max_input_length)
            with jsonlines.open(os.path.join(output_dir, 'dialogues_'+str(i)+'.json'), 'w') as writer:
                writer.write_all(dialogues)
        else:
            print('WARNING: dialogues has a len of 0')


def text_to_dialogues(text_data_dir, output_dir):
    list_of_textfiles = glob.glob(text_data_dir+'/*.txt')
    for textfile in list_of_textfiles:
        save_text_as_dialogues(textfile, os.path.join(output_dir, os.path.basename(textfile).split('.txt')[0]))


def dialogues_to_groups(plays_dialogue_dir, output_dir, tokenizer, max_input_length=1024):
    plays = glob.glob(plays_dialogue_dir + '/*')
    for play in plays:
        play_group_dir = os.path.join(output_dir, os.path.basename(play))
        os.makedirs(play_group_dir, exist_ok=True)
        preprocess_to_groups(play, play_group_dir, tokenizer, max_input_length=max_input_length)


def file_objects_at_folder(foldername, extension='.json'):
    '''
    Pulls all files in a folder, even if embedded several layers deep,
    and allows for extension specification, starting frame and ending frame.

    Params
    ------
    foldername: str
        directory for where the frames are located
    extension: str
        file extension of the image type, beginning with '.'
    '''

    image_files = []
    for root, dirs, files in os.walk(foldername):
        for name in files:
            if name.endswith(extension):
                image_files.append(os.path.join(root, name))
    image_files.sort()

    return image_files


def read_json_lines(jsonlines_filename):
    data = []
    with open(jsonlines_filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def add_taskname(dialogues_dir):
    dialogue_files = file_objects_at_folder(dialogues_dir, extension='.json')
    for filename in dialogue_files:
        # read the json lines
        filename_data = read_json_lines(filename)
        for line in filename_data:
            # add the task name to each json and re-save
            line['taskname'] = 'eme_seq2seq'
        with jsonlines.open(filename, 'w') as writer:
            writer.write_all(filename_data)


def get_token_counts(dialogue_jsons):
    # find highest context_length
    print('Longest context in these dialogues:', max([dialogue['context_len'] for dialogue in dialogue_jsons]))
    return dialogue_jsons


def map_tokens_and_scenes(play_dir):
    scene_dialogues = glob.glob(play_dir+'/*')
    scene_dialogues = [read_json_lines(d) for d in scene_dialogues]
    for i, dialogue_list in enumerate(scene_dialogues):
        dialogue_list = get_token_counts(dialogue_list)
        scene_dialogues[i] = dialogue_list
    dialogue_count = {'title': os.path.basename(play_dir), 'num_of_scenes': len(scene_dialogues), 'context_lengths': []}
    for scene in scene_dialogues:
        context_length = max([dialogue['context_len'] for dialogue in scene])
        dialogue_count['context_lengths'].append(context_length)
    return dialogue_count


def graph_dialogue_counts(list_of_dialogue_counts):
    list_of_dialogue_counts = sorted(list_of_dialogue_counts, key=lambda d: sum(d['context_lengths']))
    titles = [d['title'] for d in list_of_dialogue_counts]
    context_lengths = [sum(d['context_lengths']) for d in list_of_dialogue_counts]
    plt.bar(range(len(titles)), context_lengths)
    plt.xticks(range(len(titles)), titles, rotation=45)
    plt.savefig('test_plot.png')


def get_train_val_split(list_of_dialogue_counts, split_proportion=0.8):
    ''' Run sort on sum of context_length first, for the list_of_dialogue_counts
    '''
    context_lengths = [sum(d['context_lengths']) for d in list_of_dialogue_counts]
    total_context_lengths = sum(context_lengths)
    train_goal = int(total_context_lengths * split_proportion)
    backwards_indices = list(range(len(context_lengths)))
    backwards_indices.sort(reverse=True)
    count_to_goal = 0
    train_indices = []
    for i in backwards_indices:
        l = context_lengths[i]
        count_to_goal += l
        train_indices.append(i)
        if count_to_goal >= train_goal:
            break
    val_indices = [i for i in backwards_indices if i not in train_indices]
    return train_indices, val_indices


def create_train_and_val_directories(data_directory, output_dir):
    list_of_plays = glob.glob(data_directory+'/*')
    list_of_dialogue_counts = []
    for play in list_of_plays:
        dialogue_count = map_tokens_and_scenes(play)
        list_of_dialogue_counts.append(dialogue_count)
    list_of_dialogue_counts = sorted(list_of_dialogue_counts, key=lambda d: sum(d['context_lengths']))
    train_indices, val_indices = get_train_val_split(list_of_dialogue_counts)
    train_titles = [list_of_dialogue_counts[i]['title'] for i in train_indices]
    val_titles = [list_of_dialogue_counts[i]['title'] for i in val_indices]
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    for k, fpath in enumerate(list_of_plays):
        for title in train_titles:
            if title in fpath:
                shutil.copytree(fpath, os.path.join(train_dir, title), dirs_exist_ok=True)
        for title in val_titles:
            if title in fpath:
                shutil.copytree(fpath, os.path.join(val_dir, title), dirs_exist_ok=True)


def clean_up_context_len(data_directory):
    '''stupid Nemo can't handle extra fields at all,
    so they need to be removed. Ideally this would be
    done in the dataloading scripts that they are using,
    but everything is so abstracted as to be a pain to
    modify. 
    TODO: re-build the NeMo specific models and 
    dataloading so that this can be modified there instead
    of here.
    '''
    all_files = file_objects_at_folder(data_directory, extension='.json')
    for f in all_files:
        list_of_lines = read_json_lines(f)
        for i, line in enumerate(list_of_lines):
            # if context_len in there, delete and resave
            if 'context_len' in line.keys():
                line.pop('context_len')
            if 'context_lengths' in line.keys():
                line.pop('context_lengths')
            list_of_lines[i] = line
        with jsonlines.open(f, 'w') as writer:
            writer.write_all(list_of_lines)


def get_total_training_tokens(train_dir, tokenizer):
    all_files = file_objects_at_folder(train_dir, extension='.json')
    total_tokens = 0
    for f in all_files:
        datalines = read_json_lines(f)
        for line in datalines:
            context_length = len(tokenizer(line['context'])['input_ids'])
            response_length = len(tokenizer(line['response'])['input_ids'])
            line_tokens = context_length + response_length
            total_tokens += line_tokens
    return total_tokens


def get_total_dataset_tokens(dialogue_dir, tokenizer):
    all_files = file_objects_at_folder(dialogue_dir, extension='.json')
    total_tokens = 0
    for f in all_files:
        datalines = read_json_lines(f)
        for line in datalines:
            length = len(tokenizer(line['character_and_line'])['input_ids'])
            total_tokens += length
    return total_tokens


def main(data_text_directory, dialogue_dir, grouped_dir, splits_dir, max_input_length=1024):
    ''' Just to show how to run all the preprocessing
    together to get the final prepared dataset for NeMo
    '''
    text_to_dialogues(data_text_directory, dialogue_dir)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    dialogues_to_groups(dialogue_dir, grouped_dir, tokenizer, max_input_lenghth=max_input_length)
    create_train_and_val_directories(grouped_dir, splits_dir)
    clean_up_context_len(splits_dir)

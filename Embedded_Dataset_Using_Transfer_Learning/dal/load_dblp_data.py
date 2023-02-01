import numpy as np
import pickle as pkl

def load_preprocessed_dataset(file_path='../dataset/dblp_preprocessed_dataset.pkl'):
    with open(file_path, 'rb') as f:
        dataset = pkl.load(f)
    return dataset

def load_ae_dataset(file_path='../dataset/ae_dataset.pkl'):
    with open(file_path, 'rb') as f:
        dataset = pkl.load(f)
    return dataset

def nn_m2v_dataset_generator(model_path='../dataset/embedding_dict_gat.pkl', dataset=None, output_file_path='../dataset/ae_e_m2v_tSkill_dataset.pkl', mode='skill'):
    model = pkl.load(open(model_path, 'rb'))
    m2v_dataset = []
    counter = 1
    for record in dataset:
        id = record[0]
        ###### if mode user then output set of authors for each paper will be embedded ########
        if mode.lower() == 'user':
            try:
                skill_vec = record[1].todense()
                team_idx = record[2].todense().nonzero()[1]
                team_vec = []
                for user_id in team_idx:
                    team_vec.append(model['user'][user_id])
                m2v_dataset.append([id, skill_vec, np.sum(team_vec, axis=0)])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        ###### if mode skill then input set of skill for each paper will be embedded ########
        elif mode.lower() == 'skill':
            try:
                skill_idx = record[1].todense().nonzero()[1]
                skill_vec = []
                for skill_id in skill_idx:
                    skill_vec.append(model['skill'][skill_id])
                team_vec = record[2].todense()
                m2v_dataset.append([id, np.sum(skill_vec,axis=0), team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        ###### if mode full then both input set of skills and output set of authors for each paper will be embedded ########
        elif mode.lower() == 'full':
            try:
                skill_idx = record[1].todense().nonzero()[1]
                skill_vec = []
                for skill_id in skill_idx:
                    skill_vec.append(model['skill'][skill_id])
                team_idx = record[2].todense().nonzero()[1]
                team_vec = []
                for user_id in team_idx:
                    team_vec.append(model['user'][user_id])
                m2v_dataset.append([id, np.sum(skill_vec, axis=0), np.sum(team_vec, axis=0)])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
    with open(output_file_path, 'wb') as f:
        pkl.dump(m2v_dataset, f)

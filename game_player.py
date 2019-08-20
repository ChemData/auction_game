import os
import json
import itertools
import re
from copy import deepcopy
import numpy as np
import scipy.stats
import pandas as pd
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt
import keras as ks


class TrainingGames:
    """Play identical AIs against each other to generate training data."""

    def __init__(self, base_folder):
        self.base_folder = base_folder
        os.makedirs(self.base_folder, exist_ok=True)

    def run_games(self, num_blocks, games_per_block, head_node, model, submodel_nums, params,
                  params_to_store, explorations):
        """Run games to generate training data.

        Args:
            num_blocks (int): Number of blocks of games to run. Each block of games is
                stored together.
            games_per_block (int): Number of games in each block.
            head_node (Node class name): Name of a Node class to use as the head.
            model (NNModel): The model to use for play prediction.
            submodel_nums (tuple, None): Numbers of the submodels to use. If None, will
                use a randomly initialized submodels instead.
            params (dict): Parameters to give to the Node to create a new head.
            params_to_store (dict): Parameter values that should be stored in the run
                info file. Keys are the param name. Values are format strings of the form:
                '{param}' with any extra formating included e.g. '{param.__name__}'.
            explorations (int): Number of explorations to perform for each move.
        """
        model = model(self.base_folder)
        if submodel_nums is None:
            model._create_new(save=True)
        else:
            model.load_model(submodel_nums)
        for b_count in range(num_blocks):
            print(f'Block {b_count}')
            self.block_data = {}
            for g_count in range(games_per_block):
                print(f'\tGame {g_count}')
                head = head_node(model=model, **params)
                self._add_data(self._play_game(head, explorations))
            self._store_block(locals())

    def _play_game(self, head_node, explorations):
        """Run a game from the provided head node."""
        while not head_node.game_state.is_finished:
            head_node.do_explorations(explorations)
            head_node, _ = head_node.make_move(1)
        # The final head_node does not contain any usable data
        return head_node.parent.output_data(head_node.game_state.winner)

    def _add_data(self, new_data):
        """Add new data from a game to the data for that block."""
        self.block_data = Node.merge_arrays(self.block_data, new_data)

    def _store_block(self, params):
        """Store new training data and info from the just completed block of games."""
        data_folder = os.path.join(self.base_folder, 'data')
        os.makedirs(data_folder, exist_ok=True)

        try:
            old_info = pd.read_csv(os.path.join(self.base_folder, 'training_game_info.txt'), index_col=0)
        except FileNotFoundError:
            old_info = pd.DataFrame()
        new_info = pd.DataFrame()
        new_info['games'] = [params['games_per_block']]
        for i, id in enumerate(params['model'].ids):
            new_info[f'submodel {i}'] = [id]
        new_info['explorations'] = [params['explorations']]
        for param, fstring in params['params_to_store'].items():
            new_info[param] = [fstring.format(param=params['params'][param])]
        all_info = old_info.append(new_info, ignore_index=True, sort=False)
        all_info.to_csv(os.path.join(self.base_folder, 'training_game_info.txt'))
        np.savez(os.path.join(data_folder, str(all_info.index.values[-1])), **self.block_data)


class TrialGames:
    """Pit two AIs against each other to see which has better performance."""

    def __init__(self, base_folder):
        self.base_folder = base_folder
        os.makedirs(self.base_folder, exist_ok=True)
        self._load_results()

    def run_games(self, num_games, head_node, model, submodel_nums_sets, params,
                  explorations, game_state_gen):
        """Run games to generate training data.

        Args:
            num_games(int): Number of games to play.
            head_node (Node class name): Name of a Node class to use as the head.
            model (NNModel): The model to use for play prediction.
            submodel_nums_sets (tuple): Numbers of models to use. Each element in the
                tuple is a tuple of ints.
            params (dict): Parameters to give to the Node to create a new head.
            explorations (int): Number of explorations to perform for each move.
            game_state_gen (func): A function which returns newly initialized game
                states with the desired starting player.
        """
        model1 = model(self.base_folder)
        model2 = model(self.base_folder)
        model1.load_model(submodel_nums_sets[0])
        model2.load_model(submodel_nums_sets[1])

        results = pd.DataFrame()
        for g_count in range(num_games):
            new_info = pd.DataFrame(index=[0])
            print(f'Game {g_count}')
            if g_count < num_games//2:
                new_info['first player'] = 0
                params['game_state'] = game_state_gen(0)
                winner, turns = self._play_game(head_node, model1, model2, params, explorations)
            else:
                new_info['first player'] = 1
                params['game_state'] = game_state_gen(1)
                winner, turns = self._play_game(head_node, model2, model1, params, explorations)
            new_info['winner'] = winner
            new_info['length'] = turns
            results = results.append(new_info, ignore_index=True)
        self._add_results(results, submodel_nums_sets)

    def _play_game(self, head_node, p0_model, p1_model, params, explorations):
        """Run a game from the provided head node.

        Args:
            head_node (Node): Class name for the first node.
            p0_model (BasicNeuralNetwork): Model for AI0.
            p1_model (BasicNeuralNetwork): Model for AI1.
            params (dict): Parameters needed to initialize the head node.
            explorations (int): Number of explorations to perform before deciding a move.
        """
        models = [p0_model, p1_model]
        head = head_node(model=p0_model, **params)
        count = 1
        while not head.game_state.is_finished:
            head.do_explorations(explorations)
            head, _ = head.make_move(1)
            params['game_state'] = head.game_state
            head = head.__class__(model=models[count % 2], **params)
            count += 1

        winner = head.game_state.winner
        return winner, count-1

    def _add_results(self, results, submodels_nums):
        """Add the results of played trial games to the database."""
        try:
            new_id = self.set_info.index[-1] + 1
        except IndexError:
            new_id = 0
        results['set'] = new_id
        self.results = self.results.append(results, ignore_index=True, sort=False)
        if len(self.set_info) == 0:
            index = pd.MultiIndex.from_product([list(range(len(submodels_nums))),
                                                list(range(len(submodels_nums[0])))],
                                               names=['ai', 'sub'])
            self.set_info = pd.DataFrame(
                np.array(submodels_nums).flatten().reshape((1, -1)), columns=index)
        else:
            self.set_info.loc[new_id] = tuple(np.array(submodels_nums).flatten())
        self._save_results()

    def _load_results(self):
        """Load results from previously played games."""
        try:
            self.set_info = pd.read_csv(
                os.path.join(self.base_folder, 'trial_sets.txt'), index_col=0, header=[0, 1])
        except FileNotFoundError:
            self.set_info = pd.DataFrame()

        try:
            self.results = pd.read_csv(os.path.join(self.base_folder, 'trial_game_results.txt'))
        except FileNotFoundError:
            self.results = pd.DataFrame(columns=['set', 'winner', 'first player', 'length'])

    def _save_results(self):
        """Save results from played games."""
        self.set_info.to_csv(os.path.join(self.base_folder, 'trial_sets.txt'), index=True)
        self.results.to_csv(os.path.join(self.base_folder, 'trial_game_results.txt'), index=False)


class Node:
    id_iter = itertools.count()

    def __init__(self, model, game_state, parent, probability, exp_const, node_dict={}, is_head=False):
        """
        Args:
            model (placholder): The neural network used for decision making.
            game_state (GameState): Class that stores information about the state of the
                game.
            player (int): Number of the active player.
            parent (Node): The parent node of this node.
            probability (float): Probability of moving to this node.
            exp_const (func): A function which returns the exploration constant to use.
            node_dict (dict): Dictionary that defines what Node class is used for each
                game state.
        """
        self.id_count = self.id_iter.__next__()

        self.model = model
        self.game_state = game_state
        self.parent = parent
        self.exp_const = exp_const
        self.node_dict = node_dict
        self.is_head = is_head
        self.children = []
        self.child_states = np.array([])
        self.child_probs = np.array([])
        self.winner = None
        self._check_if_finished()
        self._assign_final_value()

        self.p = probability
        self.N = 0
        self.W = 0
        self.Q = 0

        try:
            self.depth = self.parent.depth + 1
        except AttributeError:
            self.depth = 0

    def __repr__(self):
        try:
            out = f'Node {self.id_count} (Parent: {self.parent.id_count})'
        except AttributeError:
            out = f'Node {self.id_count} (Parent: None)'
        return out

    def _find_leaf(self):
        """Find a leaf from this node."""
        if self.winner is not None:
            return self
        if len(self.children) == 0:
            return self
        return self._child_to_explore()._find_leaf()

    def _child_to_explore(self):
        """Determine which child should be explored to find a leaf."""
        child_vals = self.child_explore_vals
        options = np.where(child_vals == child_vals.max())[0]
        return self.children[np.random.choice(options)]

    @property
    def child_explore_vals(self):
        """Value of choosing its children in an MCTS leaf exploration."""
        if len(self.children) == 0:
            raise ValueError('This node has no child_explore_vals because it has no children')
        qs = np.array([c.Q for c in self.children])
        ps = np.array([c.p for c in self.children])
        ns = np.array([c.N for c in self.children])

        return (qs + self.exp_const(self.depth) * ps * np.sqrt(ns.sum())/(1 + ns)) - \
               (1 - np.array(self.allowed_moves))

    def _expand(self):
        """Expand the children of this node."""
        assert self.game_state.allowed
        if self.winner is None:
            self.v, self.child_probs = self.model.predict(self.game_state.state_array, self.__class__)
            self._extra_calcs()
            self.child_states = self.new_states()
            self._cull_disallowed_states()
            self._child_prob_normalization()
            for i, state in enumerate(self.child_states):
                type_of_node = self.node_dict.get(state.phase)
                self.children += [type_of_node(self.model, state, self, self.child_probs[i],
                                               self.exp_const, self.node_dict)]

        self._backup(self.v)

    def _extra_calcs(self):
        """Perform any follow up calculations using the NN output to compute child
        probabilities."""
        pass

    def _cull_disallowed_states(self):
        """Set the probability of any state which is not allowed to 0."""
        self.child_probs *= self.allowed_moves

    def _child_prob_normalization(self):
        """Normalize the probabilities of choosing the different children so that the
        total is 1."""
        self.child_probs /= self.child_probs.sum()

    def new_states(self):
        """Determine the new board states for each child."""
        return []

    def _check_if_finished(self):
        """Determine if the game is completed and who the winner (if any) is."""
        self.winner = self.game_state.winner

    def _assign_final_value(self):
        """If the game is concluded, determine the final value of the game."""
        if self.winner is None:
            self.v = None
        if self.winner == self.game_state.active_player:
            self.v = 1
        elif self.winner == self.game_state.other_player:
            self.v = 0
        else:
            self.v = .5

    def _backup(self, v):
        """This node was recently expanded and so its values need to be backed up to its parents."""
        self.N += 1
        self.W += v
        self.Q = self.W / self.N
        if not self.is_head:
            self.parent._backup(1 - v)

    def do_explorations(self, n):
        """Do a certain number of MCTS explorations from this node."""
        for i in range(n):
            leaf = self._find_leaf()
            leaf._expand()

    def pi(self, tau):
        """Compute the value of pi for each child."""
        ns = np.array([child.N for child in self.children]) ** (1 / tau)
        ns *= self.allowed_moves
        return ns / np.sum(ns)

    @property
    def allowed_moves(self):
        return [c.allowed for c in self.child_states]

    @property
    def children_stats(self):
        """Return a helpful pd.DataFrame which contains information about the Node's
        children."""
        output = pd.DataFrame()
        output['N'] = [c.N for c in self.children]
        output['Q'] = [c.Q for c in self.children]
        output['p'] = [c.p for c in self.children]
        output['allowed'] = self.allowed_moves
        output['explore val'] = self.child_explore_vals
        return output

    def make_move(self, tau):
        """Determine which child is best based on MCTS visits and make that child the head."""
        pi = self.pi(tau)
        move = np.random.choice(range(len(pi)), p=pi)
        new_head = self.children[move]
        new_head.is_head = True
        return new_head, move

    def output_data(self, winner):
        """Return the data from this node and all its predecessors to use for training the NN.
        """
        inp = self.game_state.state_array[np.newaxis]
        out = self._target_data(winner)[np.newaxis]
        phase = self.game_state.phase
        new_data = {f'{phase}_inp': inp, f'{phase}_out': out}
        if self.parent is None:
            return new_data
        else:
            top_dict = self.parent.output_data(winner)
            return self.merge_arrays(top_dict, new_data)

    def _target_data(self, winner):
        """Return the target (i.e. , y) data from this node based on who won."""
        return np.append(self._win_value(winner), self.pi(1))

    def _win_value(self, winner):
        """Return the value (to the active player) of the winner winning the game."""
        if winner == self.game_state.active_player:
            return 1
        if winner == self.game_state.other_player:
            return 0
        else:
            return .5

    @staticmethod
    def merge_arrays(dict1, dict2):
        """Takes two dictionaries with arrays and concatenates any arrays which have the
        same key. If there is a key in dict2 which is not in dict1, it will also be
        included as is."""
        dict1 = dict1.copy()
        for k in dict2.keys():
            try:
                dict1[k] = np.concatenate([dict1[k], dict2[k]])
            except KeyError:
                dict1[k] = dict2[k].copy()
        return dict1

    @staticmethod
    def other_player(player_num):
        """Return the number of the other player. If it is a tie, return the same."""
        return abs(player_num - 1)


class BasicNeuralNetwork:
    batch_size = 2000  # How many data points are in each batch
    epoch_size = 1000  # How many batches are in each epoch
    validation_frac = .2 # What percentage of the data to use for validation.
    loss_names = {0: ['move_choice_loss', 'win_prob_loss']}  # Names of the losses for each submodel
    output_divisions = {0: (1,)} # Where the data for the submodels is broken up for the different outputs.
    b = ks.backend.variable(1)

    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.model_folder = os.path.join(self.base_folder, 'models')
        self.data_folder = os.path.join(self.base_folder, 'data')
        self.num_submodels = len(self.loss_names.keys())
        self.ids = [0] * self.num_submodels
        os.makedirs(self.base_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)
        self.all_loss_names = deepcopy(self.loss_names)
        for i in self.loss_names.keys():
            self.all_loss_names[i] += ['loss']
            self.all_loss_names[i] += ['val_' + x for x in self.all_loss_names[i]]
        # These are not the loss weights that get updated during training but are rather
        # the even weights to use when beginning to train a model.
        self.loss_weights = {x: [ks.backend.variable(1) for y in
                                 range(len(self.loss_names[x]))] for x in
                             self.loss_names.keys()}

    def _create_new(self, save=False):
        inp = ks.Input((6, 7, 2))
        x = ks.layers.Flatten()(inp)
        x = ks.layers.Dense(50,
                            activation='relu',
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros')(x)
        x = ks.layers.Dense(50,
                            activation='relu',
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros')(x)
        out1 = ks.layers.Dense(1,
                               activation='relu',
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               name='win_prob')(x)

        out2 = ks.layers.Dense(7,
                               activation='softmax',
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               name='move_choice')(x)

        model = ks.Model(inputs=inp, outputs=[out1, out2])
        model.compile('RMSprop',
                      ['categorical_crossentropy', 'mean_squared_error'],
                      loss_weights=self.loss_weights[0])
        self.submodels = [model]

        if save:
            os.makedirs(os.path.join(self.model_folder, f'model 0'), exist_ok=True)
            num = self._newest_model(0) + 1
            self.submodels[0].save(os.path.join(self.model_folder, 'model 0', f'{num}.hdf5'))
            new_info = pd.DataFrame([[self._max_set(0)+1, 1, 'random start', 0]],
                                    columns=['set', 'epoch', 'base_model_id', 'model_type'])
            self._add_model_info(new_info, 0)

    def _load_and_merge(self, blocks=None):
        """Take all the saved game data and convert into a single dataset.
        Args:
            blocks (int or None): If int, will only load that many blocks of data. If
                None, will load all.
        """
        first = 0
        if blocks is not None:
            first = self._last_block - blocks

        self.data = {}
        for num in range(first, self._last_block):
            arrays = dict(np.load(os.path.join(self.data_folder, '{}.npz'.format(num))))
            self.data = Node.merge_arrays(self.data, arrays)

        self._split_outputs()

    def _split_outputs(self):
        """Split the targets (output) of the training data into multiple arrays to account
        for the multiple sets of outputs for each submodel."""
        for set in self.data.keys():
            split_set = []
            if '_out' in set:
                num = int(set.rstrip('_out'))
                data = self.data[set]
                prev_split = 0
                for split in self.output_divisions.get(num, []):
                    split_set += [data[:, prev_split:split]]
                    prev_split = split
                split_set += [data[:, prev_split:]]
                self.data[set] = split_set

    def train_on_data(self, epoch_count=10, submodel_ids=None, blocks=None):
        """Train a new model starting with an existing one.

        Args:
            epoch_count (int): Number of epochs of training to perform.
            submodel_ids (int, tuple or None): Which existing submodels to use as a starting
                point. If None, will use the most recent stored submodels. If tuple, will load
                the provided number for the given submodels.
            blocks (int or None): Number of blocks of data to load. If None, will load all
                available.
        """
        self._load_and_merge(blocks)

        to_use = self._newest_model
        if to_use == -1:
            self._create_new(save=True)
            to_use = 0
        if submodel_ids is not None:
            to_use = submodel_ids
        self.load_model(to_use)
        group_numbers = self._newest_model(0) + 1
        self._load_info()
        set = self.info[0].iloc[-1]['set'] + 1

        for i, model in enumerate(self.submodels):
            self.x = self.data[f'{i}_inp']
            self.y = self.data[f'{i}_out']
            self._divide_data()
            form_string = os.path.join('{}-{}.hdf5'.format(group_numbers, '{epoch}'))
            filename = os.path.join(self.model_folder, f'model {i}', form_string)
            self.history = model.fit_generator(
                self._data_generator(),
                validation_data=(self.x_val, self.y_val),
                steps_per_epoch=self.epoch_size,
                epochs=epoch_count,
                callbacks=[ks.callbacks.ModelCheckpoint(filename),
                           ScaleLosses(model.loss_weights, self.loss_names[i])])
            new_info = pd.DataFrame(index=range(epoch_count))
            new_info['set'] = set
            new_info['epoch'] = range(1, epoch_count + 1)
            new_info['base_model_id'] = to_use[i]
            new_info['batch_size'] = self.batch_size
            new_info['epoch_size'] = self.epoch_size
            new_info['loss'] = self.history.history['loss']
            for loss in self.all_loss_names[i]:
                new_info[loss] = self.history.history[loss]
            self._add_model_info(new_info, i)
        self._adjust_model_names()

    def _adjust_model_names(self):
        """Adjust model names so that those with the format [type]-[group]-[epoch] becomes just
        [type]-[ID]."""
        unadj = r'^\d+-\d+.hdf5'
        for i in range(self.num_submodels):
            folder = os.path.join(self.model_folder, f'model {i}')
            to_fix = [os.path.basename(x) for x in os.listdir(folder)]
            to_fix = [x for x in to_fix if re.match(unadj, x)]
            for f in to_fix:
                nums = re.findall(r'\d+', f)
                new_name = f'{int(nums[0]) + int(nums[1]) - 1}.hdf5'
                os.rename(os.path.join(folder, f), os.path.join(folder, new_name))

    def load_model(self, ids):
        """Load a saved model."""
        if type(ids) == int:
            ids = [ids] * self.num_submodels
        self.submodels = []
        for model_num, model_id in enumerate(ids):
            new_submodel = ks.models.load_model(os.path.join(
                self.model_folder, f'model {model_num}', f'{model_id}.hdf5'))
            # recompile to allow for loss_weights adjustment during training
            new_submodel.compile(new_submodel.optimizer, new_submodel.loss,
                                 loss_weights=self.loss_weights[model_num])

            self.submodels += [new_submodel]

    def _divide_data(self):
        """Divide data in to training and validation sets."""
        choices = np.random.randint(0, self.x.shape[0] - 1, int(self.x.shape[0]*self.validation_frac))
        self.x_train = np.delete(self.x, choices, axis=0)
        self.x_val = np.take(self.x, choices, axis=0)
        self.y_train = [np.delete(y, choices, axis=0) for y in self.y]
        self.y_val = [np.take(y, choices, axis=0) for y in self.y]

    def _data_generator(self):
        while True:
            choices = np.random.randint(0, self.x_train.shape[0] - 1, self.batch_size)
            yield np.take(self.x_train, choices, axis=0),\
                  [np.take(y, choices, axis=0) for y in self.y_train]

    def _add_model_info(self, new_info, submodel_num):
        """Add data for recently trained models into the info file."""
        self._load_info()
        self.info[submodel_num] = self.info[submodel_num].append(new_info, ignore_index=True)
        self._save_info()

    @property
    def _last_block(self):
        """Return the number of the highest data block file."""
        return max([int(os.path.basename(x).split('.')[0])
                    for x in os.listdir(self.data_folder)])

    def _newest_model(self, submodel):
        """Return the biggest model number."""
        self._load_info()
        try:
            return self.info[submodel].index[-1]
        except IndexError:
            return -1

    def _max_set(self, submodel):
        """Returns the largest set number."""
        self._load_info()
        try:
            return self.info[submodel]['set'].max()
        except KeyError:
            return -1

    def _load_info(self):
        """Load the model training results."""
        self.info = {}
        for i in range(self.num_submodels):
            path = os.path.join(self.base_folder, f'model_{i}_info.txt')
            try:
                self.info[i] = pd.read_csv(path, index_col=0)
            except (FileNotFoundError, EmptyDataError):
                self.info[i] = pd.DataFrame()

    def _save_info(self):
        """Store model training results."""
        for i in self.info.keys():
            path = os.path.join(self.base_folder, f'model_{i}_info.txt')
            self.info[i].to_csv(path)

    def loss_analysis(self):
        self._load_info()
        max_set = self.info[0].iloc[-1]['set']
        f, axs = plt.subplots(1, self.num_submodels)
        for i in range(self.num_submodels):
            data = self.info[i]
            data = data[data['set'] == max_set]
            loss_names = self.all_loss_names[i]
            loss_names = [x for x in loss_names if 'val_' not in x]
            loss_names = [a+x for x in loss_names for a in ['', 'val_']]
            for j, loss in enumerate(loss_names):
                if j % 2 == 0:
                    p = axs[i].plot(data['epoch'], data[loss], label=loss)
                else:
                    axs[i].plot(data['epoch'], data[loss], linestyle=':', label='', c=p[0].get_color())
            axs[i].legend()
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel('Loss')

        plt.show()

    def best_epochs(self, set):
        """Find the epoch(s) which have the lowest validation loss in a particular set."""
        self._load_info()
        output = {'models': [], 'epochs': []}
        for i in range(self.num_submodels):
            data = self.info[i]
            data = data[data['set'] == set]
            rolling_val = data['val_loss'].rolling(window=10, min_periods=1, center=True).mean()
            best_model = rolling_val.idxmin()
            output['models'] += [best_model]
            output['epochs'] += [data.loc[best_model, 'epoch']]

        return output


def basic_exp_const(depth):
    return 1.212


class ScaleLosses(ks.callbacks.Callback):
    """Sets the scale of different losses in a single model after a few epochs so that
    all losses are roughly comparable."""
    scale_epoch = 9  # Epoch at which to carry out scaling

    def __init__(self, loss_weights, loss_names):
        self.loss_weights = loss_weights
        self.loss_names = loss_names
        self.loss_history = np.zeros((self.scale_epoch, len(loss_weights)))

    def on_epoch_end(self, epoch, logs={}):

        if epoch < self.scale_epoch:
            for i, loss in enumerate(self.loss_names):
                self.loss_history[epoch, i] = logs[loss]

        if epoch == self.scale_epoch - 1:
            mean_loss = self.loss_history.mean(axis=0)

            adj_losses = self.loss_history[-1, :]/mean_loss/len(mean_loss)
            weights = self.loss_history[-1, :].sum()/mean_loss/len(mean_loss)/adj_losses.sum()

            for i, w in enumerate(self.loss_weights):
                ks.backend.set_value(w, weights[i])



import importlib
import os
import json
import time
import game_player
import pandas as pd



class TrainGenerations:

    def __init__(self, base_folder):
        self.base_folder = base_folder
        with open(os.path.join(self.base_folder, 'training_params.json')) as f:
            self.params = json.load(f)
        self._load_modules()
        self._format_game_params()
        self.model = eval(self.params['model'])(self.base_folder)

    def _load_modules(self):
        """Load any additional modules that are needed for the game."""

        flat_params = self.deep_values(self.params)
        for p in flat_params:
            try:
                mod = p.split('.')
                if len(mod) == 2:
                    mod_object = importlib.import_module(mod[0])
                    globals()[mod[0]] = mod_object
            except AttributeError:
                pass
            except ModuleNotFoundError:
                pass

    def _format_game_params(self):
        # Add fixed parameters
        self.params['training_games']['head_node_params']['parent'] = None
        self.params['training_games']['head_node_params']['probability'] = None
        self.params['training_games']['head_node_params']['is_head'] = True

        # Move around shared parameters
        self.params['training_games']['explorations'] = self.params['explorations']
        self.params['training_games']['num_blocks'] = self.params['blocks_per_generation']
        self.params['training_games']['model'] = eval(self.params['model'])
        self.params['training_games']['head_node'] = eval(self.params['training_games']['head_node'])
        self.params['training_games']['games_per_block'] = self.params['games_per_block']

        # Evaluate strings
        self.params['training_games']['head_node_params']['game_state'] = eval(
            self.params['training_games']['head_node_params']['game_state'])
        self.params['training_games']['head_node_params']['exp_const'] = eval(
            self.params['training_games']['head_node_params']['exp_const'])
        new_dict = {}
        for k in self.params['training_games']['head_node_params']['node_dict'].keys():
            new_dict[int(k)] = eval(
                self.params['training_games']['head_node_params']['node_dict'][k])
        self.params['training_games']['head_node_params']['node_dict'] = new_dict

    @staticmethod
    def deep_values(dictionary):
        """Return all values in a dictionary (and values of any subdictionaries)."""
        output = []
        for v in dictionary.values():
            if isinstance(v, dict):
                output += TrainGenerations.deep_values(v)
            else:
                output += [v]
        return output

    def run_generations(self, number):
        """Run a certain number of generations of training."""
        training_games = game_player.TrainingGames(self.base_folder)

        for i in range(number):
            # Identify best submodels from most recent training set
            if self.model.max_set(0) == -1:
                self.model.create_new(save=True)
            best_subs = self.model.best_epochs()['models']

            # Use these to generate new training data
            start_time = time.time()
            training_games.run_games(submodel_nums=best_subs,
                                     **self.params['training_games'])
            gen = time.time() - start_time

            # Train new models
            start_time = time.time()
            self.model.train_on_data(epoch_count=self.params['epochs_per_generation'],
                                     submodel_ids=best_subs,
                                     blocks=self.params['blocks_per_training'])
            train_time = time.time() - start_time

            # Output the metadata
            vals = {}
            vals['generation_time'] = gen
            vals['training_time'] = train_time
            for i, v in enumerate(best_subs):
                vals[f'submodel {i}'] = v

            for v in self.params['params_to_store']:
                vals[v] = self.params[v]
            self._add_generation_info(vals)

    def _add_generation_info(self, new_row):
        """Add information about the most recent generation."""
        try:
            old_data = pd.read_csv(
                os.path.join(self.base_folder, 'generation_params.txt'), index_col=0)
        except FileNotFoundError:
            old_data = pd.DataFrame()
        old_data.append(pd.Series(new_row), ignore_index=True).to_csv(
            os.path.join(self.base_folder, 'generation_params.txt'))


t = TrainGenerations('gen_training')
t.run_generations(1)

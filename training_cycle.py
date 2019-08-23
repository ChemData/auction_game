import importlib
import os
import json
import game_player



class TrainGenerations:

    def __init__(self, base_folder):
        self.base_folder = base_folder
        self._load_game_file()
        self._load_modules()

    def _load_game_file(self):
        """Load the JSON file which describes the model creation/training parameters."""
        with open(os.path.join(self.base_folder, 'training_params.json')) as f:
            p = json.load(f)

        # Add fixed parameters
        p['training_games']['head_node_params']['parent'] = None
        p['training_games']['head_node_params']['probability'] = None
        p['training_games']['head_node_params']['is_head'] = True

        # Move around shared parameters
        p['training_games']['model'] = p['model']
        p['training_games']['explorations'] = p['explorations']

        self.params = p

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

    def generate_test_games(self, submodel_nums):
        """Run head to head games to produce more training data."""
        t = game_player.TrainingGames(self.base_folder)
        num_blocks= self.params['games_per_gen']//self.params['games_per_block']
        t.run_games(num_blocks=num_blocks,
                    submodel_nums=submodel_nums,
                    **self.params['training_games'])

    def train_model(self, submodel_ids):
        m = self.params['model'](self.base_folder)
        m.train_on_data(epoch_count=self.params['epochs_per_training'],
                        submodel_ids=submodel_ids,
                        blocks=self.params['blocks_per_training'])
        return m.best_epochs()

    def initialize_starting_model(self):
        """Create a randomly initialized model to begin training with."""
        m = self.params['model'](self.base_folder)
        m.create_new(save=True)







t = TrainGenerations('gen_training')
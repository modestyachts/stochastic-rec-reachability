"""Environment objects which contain functionalty for loading and processing datasets and recommenders"""
import abc
from ast import literal_eval
import collections
import json
import numpy as np
import os
import pandas as pd
import pickle
from scipy.sparse import csr_matrix, dok_matrix
from scipy.stats import skewnorm
from time import time

# TODO remove once pip installable
import sys
sys.path.append(os.path.join(os.path.expanduser("~"), 'recsys/recsys-eval/'))
sys.path.append(os.path.join(os.path.expanduser("~"), 'recsys/RecLab/'))
from reclab import data_utils
from reclab.recommenders import LibFM, KNNRecommender
from reclab.recommenders.sparse import EASE

from config import DATAPATH, MODELPATH
from helper_utils import get_seen_mask
import model_interface
import mask_utils

REC_MODELS = ['libfm', 'knn', 'ease']
ML_DATASETS = ['ml-100k', 'ml-1m', 'ml-10m']

class Environment(metaclass=abc.ABCMeta):
    """ The abstract base class interface for environment
    """
    @property
    def name(self):
        """ Get the name of the environment """
        return self._name

    @property
    def recommender_model(self):
        return self._recommender_model

    @name.setter
    def name(self, value):
        self._name = value

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train_recommender(self):
        raise NotImplementedError

    @abc.abstractmethod
    def attach_recommender(self):
        raise NotImplementedError

    def compute_dense_predictions(self, **kwargs):
        self.dense_predictions = model_interface.get_predicted_scores(self.trained_model_dict, **kwargs)


# ========== Synthetic environment classes ============
class RandomEnvironment(Environment):
    """ The interface for randomly generated environment
    """
    def __init__(self, num_user, num_item):
        self.num_user = num_user
        self.num_item = num_item
        self._name = 'random_{}_{}'.format(num_user, num_item)

    def initialize(self, recommender_model=None, dense_params={}, sample_params={}, train_params={}):
        self.create_dense_ratings(**dense_params)
        self.sample_observed_ratings(**sample_params)
        if recommender_model is not None:
            self.train_recommender(recommender_model, **train_params)
            self._name = 'random_{}_{}_rec_{}'.format(self.num_user, self.num_item, recommender_model)

    def create_dense_ratings(self, min = 1, max = 5, kind = 'uniform', random_state = None):
        """ Samples uniformly from min-max to create random dense scores

        Parameters
        ----------
        num_user : int
        num_item : int
        kind : str, optional
            rating types, by default 'uniform'
        min_rat : int, optional
            by default 1
        max_rat : int, optional
            by default 5

        Sets
        -------
        self.dense_ratings : np array
            num_user x num_item matrix of true underlying ratings
        """
        r = np.random.RandomState(seed=random_state)
        if kind == 'uniform':
            self.dense_ratings = r.randint(low=min, high=max+1, size=(self.num_user, self.num_item))
        else:
            raise NotImplementedError("To be implemented")
        self._users = dict.fromkeys(range(self.num_user), [])
        self._items = dict.fromkeys(range(self.num_item), [])


    def sample_observed_ratings(self, density=0.1, kind='uniform', random_state=None):
        """Given a latent model, get an observed subset of ratings

        Parameters
        ----------
        full_rating_mat : array
            num_user x num_item matrix of true underlying ratings
        density : float, optional
            proportion of the observed rating matrix, by default 0.1
        kind : str, optional
            type of rating subsetting, by default 'uniform'
            'uniform': ratings are sampled uniformly from the rating matrix
            'proportional': ratings are sampled proportionally to their rating
            'popularity': items are sampled by popularity (biases)
                        users are sampled uniformly
        Sets
        ----------
        observed_ratings : sparse matrix
            num_user x num_items observed rating matrix
        """

        if kind == "uniform":
            blocked_mask = csr_matrix(([], ([], [])), shape=(self.num_user, self.num_item))
            mask = mask_utils.make_random_mask(blocked_mask, density = density, random_state = random_state)
            self.observed_ratings = csr_matrix(mask.multiply(self.dense_ratings))
            self.seen_mask = mask
        if kind == "proportional":
            r = np.random.RandomState(seed=random_state)
            rating_probs = np.square(self.dense_ratings.reshape(-1, 1))
            rating_probs = (rating_probs/sum(rating_probs)).reshape(-1)
            selected_pairs = r.choice(a=self.num_item*self.num_user,
                                    size = np.int(density*self.num_item*self.num_user),
                                    replace=False,
                                    p=rating_probs)
            selected_pairs = [np.unravel_index(idx, (self.num_user, self.num_item)) for idx in selected_pairs]
            observed_ratings = np.zeros([self.num_user, self.num_item])
            for pair in selected_pairs:
                observed_ratings[pair] = self.dense_ratings[pair]
            self.observed_ratings = csr_matrix(observed_ratings)
            self.seen_mask = csr_matrix(self.observed_ratings, dtype=bool)
        if kind == "popularity":
            raise NotImplementedError("To be implemented")
        self._observed_ratings_dict = dict(dok_matrix(self.observed_ratings))
        for key, value in self._observed_ratings_dict.items():
            self._observed_ratings_dict[key] = (value, [])

    def train_recommender(self, recommender_model, **train_params):
        """ Train a fresh recommender for the dataset

        Parameters
        ----------
        recommender_model : str
            name of recommender model to use
        train_params : kw args
            parameters sent to recommender model for training

        Sets
        -------
        self.trained_recommender : reclab.recommender
        self.trained_model_dict : dict
        """
        if recommender_model == 'libfm':
            recommender = LibFM(max_num_users=self.num_user, max_num_items=self.num_item,
                            num_user_features=0, num_item_features=0, num_rating_features=0,
                            **train_params)
        elif recommender_model == 'knn':
            recommender = KNNRecommender(user_based=False, **train_params)
        elif recommender_model == 'ease':
            recommender = EASE(**train_params)
        else:
            raise ValueError("Choose a valid recommender model: 'libfm', 'knn' or 'ease'")
        recommender.reset(self._users, self._items, self._observed_ratings_dict)
        self.trained_recommender = recommender
        self.trained_model_dict = model_interface.extract_model_parameters(self.trained_recommender)
        self._recommender_model = recommender_model

    def attach_recommender(self, recommender_model, **train_params):
        self.train_recommender(recommender_model, **train_params)

class RandomLatentEnvironment(RandomEnvironment):
    """ Random Environment class for synthetic data created
    from a latent model """
    def __init__(self, num_user, num_item, latent_dim=8):
        self.num_user = num_user
        self.num_item = num_item
        self.latent_dim = latent_dim
        self._name = 'random_dataset_{}_{}_{}'.format(num_user, num_item, latent_dim)

    def initialize(self, recommender_model=None, latent_params={}, dense_params={}, sample_params={}, train_params={}):
        self.make_latent_model(**latent_params)
        self.create_dense_ratings(**dense_params)
        self.sample_observed_ratings(**sample_params)
        if recommender_model is not None:
            self.train_recommender(recommender_model, **train_params)
            self._name = 'random_dataset_{}_{}_{}_rec_{}'.format(self.num_user, self.num_item, self.latent_dim, recommender_model)

    def make_latent_model(self,
                          user_skew=0, user_loc=1.2, user_scale=0.4,
                          item_skew=-4, item_loc=1.3, item_scale=0.7,
                          global_bias=1.2, random_state=None):
        user_factors = np.random.normal(size=[self.num_user, self.latent_dim])/np.sqrt(self.latent_dim)
        item_factors = np.random.normal(size=[self.num_item, self.latent_dim])/np.sqrt(self.latent_dim)
        user_bias =  skewnorm.rvs(0, loc=1.2, scale=0.4, size=self.num_user, random_state=np.random.RandomState(seed=random_state))
        item_bias =  skewnorm.rvs(-4, loc=1.3, scale=0.7, size=self.num_item, random_state=np.random.RandomState(seed=random_state))
        global_bias = 1.2
        self.latent_model_dict = {  "user_factors": user_factors,
                                    "user_bias": user_bias,
                                    "item_factors": item_factors,
                                    "item_bias": item_bias,
                                    "global_bias": global_bias,
                                    "latent_dim": self.latent_dim,
                                    "kind": 'libfm',
                                    }

    def create_dense_ratings(self, clip = True, min = 1, max = 5):
        self.dense_ratings = model_interface.get_predicted_scores(self.latent_model_dict, clip = clip, min_score = min, max_score = max)
        self._users = dict.fromkeys(range(self.num_user), [])
        self._items = dict.fromkeys(range(self.num_item), [])


# ========== Environment classes from rating datasets ============

class RatingEnvironment(Environment):
    """ Rating Environment classes, based on dataset file """

    def initialize(self, recommender_model = None, save = False, **train_params):
        self.load_data()
        if recommender_model is not None:
            self.attach_recommender(recommender_model, **train_params)
            if save == True:
                self.save_trained_recommender()

    @abc.abstractmethod
    def load_data(self):
        raise NotImplementedError

    @property
    def num_item(self):
        return len(self._items)

    @property
    def num_user(self):
        return len(self._users)

    @property
    def dataset_name(self):
        return self._dataset_name

    def train_recommender(self, recommender_model, modelpath=MODELPATH, **train_params,):
        """ Train a fresh recommender for the dataset

        Parameters
        ----------
        recommender_model : str
            name of recommender model to use
        modelpath : str
            location to look for tuned model parameters
        train_params : kw args
            parameters sent to recommender model for training

        Sets
        -------
        self.trained_recommender : reclab.recommender
        self.trained_model_dict : dict
        self.seen_mask : csr matrix
        self.observed_ratings : csr matrix
        self.inner_to_outer_uid : list
        self.inner_to_outer_iid : list
        """
        if recommender_model == 'ease':
            recommender_class = EASE
            default_params = {}
            param_key = 'EASE'
        elif recommender_model == 'knn':
            recommender_class = KNNRecommender
            default_params = dict(user_based=False)
            param_key = 'KNN'
        elif recommender_model == 'libfm':
            recommender_class = LibFM
            default_params = dict(num_user_features=0,
                                num_item_features=0,
                                num_rating_features=0,
                                max_num_users=self.num_user,
                                max_num_items=self.num_item,
                                method='sgd',
                                num_iter=128, # following https://arxiv.org/pdf/1905.01395.pdf
                                num_two_way_factors=64, # following https://arxiv.org/pdf/1905.01395.pdf
                                init_stdev=1.0,
                                seed=0)
            param_key = 'LibFM'
        if len(train_params) == 0:
            try:
                with open(os.path.join(modelpath, 'best_params.json'), 'r') as fp:
                    all_best_params = json.load(fp)
                train_params = all_best_params[self.dataset_name][param_key]
                del train_params['average_metric']
            except Exception:
                train_params = dict()

        default_params.update(train_params)
        recommender = recommender_class(**default_params)
        recommender.reset(self._users, self._items, self._observed_ratings_dict)  # <-- all 3 of these are in outer id
        self.trained_recommender = recommender
        self.trained_model_dict = model_interface.extract_model_parameters(self.trained_recommender)
        self._recommender_model = recommender_model
        self.seen_mask = get_seen_mask(self.trained_recommender) #inner ids
        self.observed_ratings = csr_matrix(recommender._ratings)
        self.inner_to_outer_uid = recommender._inner_to_outer_uid
        self.inner_to_outer_iid = recommender._inner_to_outer_iid

    def attach_recommender(self, recommender_model, tag='best', filename=None, modelpath=MODELPATH,
                           **train_params):
        """ Attach a recommender, either by loading or training

        Parameters
        ----------
        recommender_model : str
            name of recommender model to use
        tag : str
            tag describing the recommender, optional
        filename : str
            name specifying recommender file to load, optional
        modelpath : str
            location to look for model
        train_params : kw args
            parameters sent to recommender model for training

        Sets
        -------
        self.trained_model_dict : dict
        self.seen_mask : csr matrix
        self.observed_ratings : csr matrix
        self.inner_to_outer_uid : list
        self.inner_to_outer_iid : list
        """
        assert recommender_model in REC_MODELS
        if tag is None:
            tag = 'best'
        if filename is not None:
            if not filename.endswith('.pkl'):
                filename = "{}.pkl".format(filename)
        else:
            filename = "{}_{}_{}.pkl".format(self.dataset_name, tag, recommender_model)
        try:
            save_dict = pickle.load(open(os.path.join(modelpath, filename),"rb"))
            self._recommender_model = recommender_model
            self.seen_mask = save_dict["seen_mask"]
            self.trained_model_dict = save_dict["trained_model"]
            self.observed_ratings = save_dict["observed_ratings"]
            self.inner_to_outer_uid = save_dict["inner_to_outer_uid"]
            self.inner_to_outer_iid = save_dict["inner_to_outer_iid"]
            self.trained_recommender = None  #<- None because loaded from pickle
            print("Saved trained model loaded from {}".format(filename))
        except FileNotFoundError:
            print("did not find saved model at {}".format(os.path.join(modelpath, filename)))
            self.train_recommender(recommender_model, modelpath=modelpath, **train_params)

    def save_trained_recommender(self, tag = 'best'):
        """ Save key elements of recommender to dict for later use. """
        filename = "{}_{}_{}.pkl".format(self.dataset_name, tag, self.recommender_model)
        save_dict = {}
        save_dict['trained_model'] = self.trained_model_dict
        save_dict['seen_mask'] = self.seen_mask
        save_dict['inner_to_outer_uid'] = self.inner_to_outer_uid
        save_dict['inner_to_outer_iid'] = self.inner_to_outer_iid
        save_dict['observed_ratings'] = self.observed_ratings
        pickle.dump(save_dict, open(os.path.join(MODELPATH, filename),"wb"))
        print("saving recommender model to", "{}".format(os.path.join(MODELPATH, filename)))

    def reindex_by_inner_id(self):
        """ Reindex user and item attributes to match recommender indexing. """
        # initializing attributes if empty
        if self.user_attributes is None:
            self.user_attributes = pd.DataFrame(self._users.keys(), columns=['user_id'])
        if self.item_attributes is None:
            self.item_attributes = pd.DataFrame(self._items.keys(), columns=['item_id'])
        # resetting user/item id to a column rather than index
        if 'user_id' not in self.user_attributes.columns:
            self.user_attributes['user_id'] = self.user_attributes.index
        if 'item_id' not in self.item_attributes.columns:
            self.item_attributes['item_id'] = self.item_attributes.index
        # mapping inner to outer id
        outer_to_inner_uid_map = dict(zip(self.inner_to_outer_uid, range(self.num_user)))
        outer_to_inner_iid_map = dict(zip(self.inner_to_outer_iid, range(self.num_item))) # <- using dicts is more efficient for re-indexing
        self.user_attributes['inner_id'] = self.user_attributes.apply(lambda x: (outer_to_inner_uid_map.get(x['user_id'])), axis = 1)
        self.item_attributes['inner_id'] = self.item_attributes.apply(lambda x: (outer_to_inner_iid_map.get(x['item_id'])), axis = 1)
        self.user_attributes.dropna(subset=['inner_id'], inplace=True)
        self.item_attributes.dropna(subset=['inner_id'], inplace=True)
        assert self.num_user == len(self.user_attributes)
        assert self.num_item == len(self.item_attributes)
        # ensure that the ids are ints
        self.user_attributes.inner_id = self.user_attributes.inner_id.astype(int)
        self.item_attributes.inner_id = self.item_attributes.inner_id.astype(int)
        # setting inner id as index
        self.user_attributes.set_index('inner_id', inplace=True)
        self.item_attributes.set_index('inner_id', inplace=True)
        # sorting the dataframe by index for easy assignment from exp matrices
        self.user_attributes.sort_index(inplace=True)
        self.item_attributes.sort_index(inplace=True)

class MovieLensEnvironment(RatingEnvironment):
    def __init__(self, ml_type):
        self._name = None
        assert ml_type in ML_DATASETS
        self._dataset_name = ml_type
        self.dense_ratings = None

    def load_data(self):
        """ Load data using RecLab. """
        rating_data, user_attributes, item_attributes = data_utils.get_data(self.dataset_name, load_attributes=True)
        self.user_attributes = user_attributes
        self.item_attributes = item_attributes
        self._users, self._items, self._observed_ratings_dict = data_utils.dataset_from_dataframe(rating_data, shuffle=True) # <-- this is not really shuffling anything

class LastFMEnvironment(RatingEnvironment):
    def __init__(self):
        self._dataset_name = 'lastfm-360k'
        self._name = None
        self.dense_ratings = None

    def load_data(self, density = 0.1, random_state = 19):
        """ Load data using RecLab, then subset. """
        r = np.random.RandomState(seed=random_state)
        rating_data, user_attributes, item_attributes = data_utils.get_data(self.dataset_name, load_attributes=True)
        self.user_attributes = user_attributes
        self.item_attributes = item_attributes
        rating_data = rating_data.drop(rating_data[rating_data['rating'] == 0].index)
        user_ids = np.unique(rating_data['user_id'])
        original_num_items = len(np.unique(rating_data['item_id']))
        user_id_subset = r.choice(user_ids, size=int(density*len(user_ids)), replace=False)
        rating_data = rating_data[rating_data['user_id'].isin(user_id_subset)]

        item_ids = np.unique(rating_data['item_id'])
        item_id_subset = r.choice(item_ids, size=int(density*original_num_items), replace=False)
        rating_data = rating_data[rating_data['item_id'].isin(item_id_subset)]
        self._users, self._items, self._observed_ratings_dict = data_utils.dataset_from_dataframe(rating_data, shuffle=True)

class YouTubeEnvironment(RatingEnvironment):
    def __init__(self):
        self._dataset_name = 'youtube'
        self._name = None
        self.dense_ratings = None

    def load_data(self, datapath=DATAPATH):
        """ Load data from specified download location. """
        filename = os.path.join(datapath, 'youtube', 'IndividualLogs_dataTable.csv')
        try:
            data = pd.read_csv(filename)
        except FileNotFoundError as e:
            raise FileNotFoundError(e, "You must download YouTube datafile from " +
                  "https://github.com/sTechLab/YouTubeDurationData/blob/master/IndividualLogs_dataTable.csv",
                  str(filename))
        rating_data = pd.DataFrame({})
        rating_data['item_id'] = data['video_id']
        rating_data['user_id'] = data['user_id']
        rating_data['rating'] = np.log(data['dwell_time']+1)

        self.user_attributes = data.groupby(['user_id']).agg(
                                       gender  = ('user_gender',
                                                  lambda x: x.value_counts().index[0]),
                                       zipcode  = ('user_zip',
                                                   lambda x: x.value_counts().index[0]),
                                      )
        self.item_attributes = data.groupby(['video_id']).agg(
                                       view_count  = ('viewCount',
                                                       lambda x: x.min()),
                                       favorite_count  = ('favoriteCount',
                                                          lambda x: x.min()),
                                       dislike_count = ('dislikeCount',
                                                        lambda x: x.min()),
                                       like_count = ('likeCount',
                                                        lambda x: x.min()),
                                       duration = ('duration',
                                                        lambda x: x.mean()),
                                       publish_time = ('publishedAt',
                                                        lambda x: x.value_counts().index[0]),
                                       category = ('category_term',
                                                        lambda x: x.value_counts().index[0]),
                                       num_share = ('numShare',
                                                        lambda x: x.min()),
                                       num_subscriber = ('numSubscriber',
                                                        lambda x: x.min()),
                                       average_view_duration = ('averageViewDuration',
                                                        lambda x: x.mean()),
                                       comment_count = ('commentCount',
                                                        lambda x: x.min()),
                                      )
        self.item_attributes.index.name = 'item_id'

        self._users, self._items, self._observed_ratings_dict = data_utils.dataset_from_dataframe(rating_data, shuffle=True)

class MIND(RatingEnvironment):
    def __init__(self, size='small', kind='train'):
        assert size in ['small', 'large']
        assert kind in ['dev', 'train', 'test']
        if size == 'small':
            assert kind != 'test'

        self._name = None
        self._dataset_name = "mind_{}_{}".format(size, kind)
        self.dense_ratings = None
        self.size = size
        self.kind = kind

    def initialize(self, recommender_model = None, save = False, history = True, non_click = True, attributes=True, **train_params):
        try:
            self.load_data(history=history, non_click=non_click, attributes = attributes)
        except FileNotFoundError as e:
            raise FileNotFoundError(e, "You must download MIND data " +
                  "https://msnews.github.io/#about-mind",
                  str(filename))
        if recommender_model is not None:
            self.attach_recommender(recommender_model, **train_params)
            if save == True:
                self.save_trained_recommender()

    def _load_session_data(self, history = True, non_click = True, datapath=DATAPATH):
        """ Load news consumption data as implicit ratings. """
        self.history = history
        filepath = os.path.join(datapath, 'mind', self.size, self.kind)
        clicked_filename = os.path.join(filepath, 'clicked.csv')
        non_clicked_filename = os.path.join(filepath, 'non_clicked.csv')
        history_filename = os.path.join(filepath, 'history.csv')

        clicked_df = pd.read_csv(clicked_filename)

        if non_click:
            non_clicked_df = pd.read_csv(non_clicked_filename)
        else:
            non_clicked_df = None

        if history:
            history_df = pd.read_csv(history_filename)
        else:
            history_df = None

        rating_df = pd.concat([clicked_df, non_clicked_df, history_df])
        return rating_df

    def _load_news_data(self, datapath=DATAPATH):
        """ Load data about news items. """
        filepath = os.path.join(datapath, 'mind', self.size, self.kind)
        news_filename = os.path.join(filepath, 'processed_news.tsv')
        news_df = pd.read_csv(news_filename, sep = "\t")
        process_cols = [ "Title_Entities", "Abstract_Entities", "Wiki_List"]
        for col in process_cols:
            news_df[col]= news_df.apply(lambda x: literal_eval(x[col]), axis = 1)
        news_df.set_index('item_id')
        return(news_df)

    def load_data(self, history = True, non_click = True, attributes=True, datapath=DATAPATH):
        """ Load MIND data from specified download location. """
        rating_df = self._load_session_data(history=history, non_click=non_click, datapath=datapath)
        self._users, self._items, self._observed_ratings_dict = data_utils.dataset_from_dataframe(rating_df, shuffle=True)

        self.user_attributes = None
        self.item_attributes = None

        if attributes:
            news_df = self._load_news_data(datapath=datapath)
            self.item_attributes = news_df

class AggregateMIND(MIND):
    def __init__(self, size = 'small', kind = 'dev', agg_by = 'subcat'):
        assert agg_by in ['cat', 'subcat', 'wiki']
        super().__init__(size, kind)
        self._dataset_name = "mind_aggregate_{}_{}_{}".format(size, kind, agg_by)
        self.agg_by = agg_by

    def load_data(self, history=True, datapath=DATAPATH):
        """ Load and aggregate MIND data. """
        self.history = history
        rating_df = self._load_session_data(history=history, non_click=False, datapath=datapath)
        news_df = self._load_news_data(datapath=datapath)
        aggregator_col_name = {'cat': 'Category', 'subcat':'Subcategory', 'wiki':'Wiki_List'}
        agg_col = aggregator_col_name[self.agg_by]
        agg_dict = dict(zip(news_df['item_id'], news_df[agg_col]))
        rating_df[agg_col] = rating_df.apply(lambda x: agg_dict[x['item_id']], axis = 1)
        if self.agg_by == 'wiki':
            # each article is associated with possibly many wiki concepts
            rating_df = rating_df.explode(agg_col)

        rating_df = rating_df.groupby(['user_id',agg_col]).count()['rating'].reset_index()
        rating_df['item_id'] = rating_df[agg_col]
        rating_df['rating'] = np.log(rating_df['rating'] + 1)
        self._users, self._items, self._observed_ratings_dict = data_utils.dataset_from_dataframe(rating_df, shuffle=True)

        self.item_attributes = None
        self.user_attributes = None

    def initialize(self, recommender_model = None, save = False, history = True, **train_params):
        self.load_data(history=history)
        if recommender_model is not None:
            self.attach_recommender(recommender_model, **train_params)
            if save == True:
                self.save_trained_recommender()

    def load_wiki_vectors(self):
        """ Load wiki embeddings for MIND data. """
        filepath = os.path.join(DATAPATH, 'mind', self.size, self.kind)
        entities_filename = os.path.join(filepath, "entity_embedding.vec")
        with open(entities_filename) as f:
            lines = f.readlines()
        wiki_vec_dict = collections.defaultdict()
        for line in lines:
            line_arr = line.split('\t')
            key = line_arr[0]
            val = np.array(line_arr[1:-1]).astype(np.float)
            wiki_vec_dict[key] = val
        return wiki_vec_dict
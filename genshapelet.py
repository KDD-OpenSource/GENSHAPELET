import numpy as np
import pandas as pd
import time
import os
import sys
from copy import copy
from fastdtw import fastdtw
from scipy import interpolate
from scipy.stats import levy, zscore, mode
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import *
from scipy.spatial.distance import *
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN


def pairwise_fastdtw(X, **kwargs):
    X = [list(enumerate(pattern)) for pattern in X]
    triu = [fastdtw(X[i], X[j], **kwargs)[0] if i != j else 0 for i in range(len(X)) for j in range(i, len(X))]

    matrix = np.zeros([len(X)] * 2)
    matrix[np.triu_indices(len(X))] = triu
    matrix += np.tril(matrix.T, -1)

    return matrix


class individual:
    def __init__(self, start: list = None, slen: list = None):
        if start is None:
            start = []
        if slen is None:
            slen = []
        self.start = start
        self.slen = slen
        self.cluster = None


class genshapelet:
    def __init__(self, ts_path: 'path to file', nsegments, min_support, smin, smax, output_folder=''):
        self.ts = pd.read_csv(ts_path, header=None)
        self.ts_path = ts_path
        self.nsegments = nsegments
        if self.nsegments is None:
            self.nsegments = int(len(self.ts) / (2 * smax) + 1)
        if self.nsegments < 2:
            sys.exit('nsegments must be at least 2 for computing clustering quality')
        self.min_support = min_support
        self.smin = smin
        self.smax = smax
        if os.path.exists(output_folder):
            pass
        elif os.access(output_folder, os.W_OK):
            pass
        else:
            sys.exit('output_folder not createable.')
        self.output_folder = output_folder
        self.probability = 2 / self.nsegments
        self.random_walk = False

    def run(self, popsize: dict(type=int, help='> 3, should be odd'), sigma: dict(type=float, help='mutation factor'),
            t_max, pairwise_distmeasures=[
                (pairwise_distances, {'metric': 'cosine'}),
                (pairwise_distances, {'metric': 'chebyshev'}),
                (pairwise_distances, {'metric': 'euclidean'}),
                (pairwise_fastdtw, {'dist': euclidean})],
            fusion=True, notes=''):

        # For pairwise_distances
        # From scikit-learn: ['cityblock', 'cosine', 'euclidean', pairwise_distancesl1', 'l2', 'manhattan'].
        # From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming',
        # 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        # 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

        print('-->')
        # print('working with ' + str(self.nsegments) + ' windows')

        t_max *= 60
        t_start = time.time()

        population, fitness = [], []
        for i in range(0, popsize):
            population.append(self.make_individual())
            fitness.append(self.evaluate(population[i], pairwise_distmeasures, fusion))

        fitness_curve = []

        best_fit = -np.inf
        iterations = 0
        t_elapsed = time.time() - t_start
        while(t_elapsed < t_max):
            order = np.argsort(fitness)[::-1]
            ix_maxfitness = order[0]
            if(fitness[ix_maxfitness] > best_fit):
                best_fit = fitness[ix_maxfitness]
                fitness_curve.append((t_elapsed, iterations, best_fit))
                # print((t_elapsed, iterations, best_fit))

            # if(iterations % 500) == 0:
            #    print((t_elapsed, iterations, best_fit))

            new_population = []
            new_population.append(population[ix_maxfitness])  # elite
            fitness[0] = fitness[ix_maxfitness]

            if self.random_walk:
                for i in range(1, popsize):
                    new_population.append(self.make_individual())
                    fitness[i] = self.evaluate(population[i], pairwise_distmeasures, fusion)
            else:
                for i in range(1, int(popsize / 2), 2):
                    new_population.append(population[order[i]])
                    new_population.append(population[order[i + 1]])
                    self.crossover(new_population[i], new_population[i + 1])
                    self.mutate(new_population[i], sigma)
                    self.mutate(new_population[i + 1], sigma)
                    fitness[i] = self.evaluate(new_population[i], pairwise_distmeasures, fusion)
                    fitness[i + 1] = self.evaluate(new_population[i + 1], pairwise_distmeasures, fusion)
                for i in range(int(popsize / 2), popsize):
                    new_population.append(self.make_individual())
                    fitness[i] = self.evaluate(new_population[i], pairwise_distmeasures, fusion)

            population = new_population

            iterations += 1
            t_elapsed = time.time() - t_start

        ix_maxfitness = np.argmax(fitness)
        fitness_curve.append((t_max, iterations, fitness[ix_maxfitness]))

        # print('t_elapsed: ' + str(t_elapsed))
        # print('iterations: ' + str(iterations))
        # print('fitness: ' + str(fitness[ix_maxfitness]))

        name = self.make_filename(popsize, sigma, t_max, notes)
        self.write_shapelets(population[ix_maxfitness], name)
        self.write_fitness(fitness_curve, name)

        # print(population[ix_maxfitness].start)
        # print(population[ix_maxfitness].slen)
        # print(population[ix_maxfitness].cluster)
        # print(self.evaluate(population[ix_maxfitness], pairwise_distmeasures, fusion))
        print('--<')

        return 0

    def evaluate(self, x: individual, pairwise_distmeasures, fusion):
        # get patterns from individual
        patterns, classlabels = [], []
        for i in range(len(x.start)):
            df = self.ts.loc[x.start[i]:x.start[i] + x.slen[i] - 1, :]
            classlabels.append(mode(df.loc[:, [0]])[0][0][0])  # xD
            df = df.loc[:, [1]].apply(zscore).fillna(0)  # consider extending for multivariate ts

            upsampled_ix = np.linspace(0, len(df) - 1, self.smax)  # upsampling
            new_values = interpolate.interp1d(np.arange(len(df)), np.array(df).flatten(), kind='cubic')(upsampled_ix)
            patterns.append(new_values)
        patterns = np.array(patterns)
        classlabels = np.array(classlabels)
        # print('patterns\n' + str(patterns))  # DEBUG
        # print('classlabels ' + str(classlabels))  # DEBUG

        distances = {}
        cols = len(patterns)
        for measure, params in pairwise_distmeasures:
            distances[str(measure) + str(params)] = measure(patterns, **params)[np.triu_indices(cols)]
        distances = pd.DataFrame(distances)

        if fusion:
            clf = LogisticRegression()
            different_class = np.zeros([cols] * 2)
            different_class[classlabels[:, None] != classlabels] = 1
            different_class = different_class[np.triu_indices(cols)]

            if 1 in different_class:
                clf.fit(distances, different_class)
                combined_distance = clf.predict_proba(distances)[:, 1]
            else:
                return -np.inf

            dist_matrix = np.zeros([cols] * 2)
            dist_matrix[np.triu_indices(cols)] = combined_distance
            dist_matrix += np.tril(dist_matrix.T, -1)
        else:
            measure, params = pairwise_distmeasures[0]
            dist_matrix = measure(patterns, **params)
        # print('dist_matrix\n' + str(dist_matrix))  # DEBUG

        # epsilon! consider: eps=dist_matrix.mean()/1.5
        db = DBSCAN(eps=dist_matrix.mean(), min_samples=self.min_support, metric='precomputed', n_jobs=-1).fit(dist_matrix)
        x.cluster = db.labels_

        try:
            fitness = silhouette_score(dist_matrix, x.cluster)
        except Exception as e:
            fitness = -np.inf

        # print(fitness)  # DEBUG
        return fitness

    def validate(self, x):
        order = np.argsort(x.start)
        for i in range(len(order)):
            for j in range(1, len(order) - i):
                if(x.start[order[i + j]] - x.start[order[i]] > self.smax):
                    break
                if(x.start[order[i]] + x.slen[order[i]] > x.start[order[i + j]]):
                    return False
        return True

    def mutate(self, x, sigma):
        for i in range(len(x.start)):
            if(np.random.uniform() < self.probability):
                tmp_start, tmp_slen = copy(x.start[i]), copy(x.slen[i])
                x.slen[i] += int(sigma * (self.smax + 1 - self.smin) * levy.rvs())
                x.slen[i] = (x.slen[i] - self.smin) % (self.smax + 1 - self.smin) + self.smin
                x.start[i] = (x.start[i] + int(sigma * len(self.ts) * levy.rvs())) % (len(self.ts) - x.slen[i])
                if not self.validate(x):
                    x.start[i], x.slen[i] = copy(tmp_start), copy(tmp_slen)
        return 0

    def crossover(self, x, y):
        for i in range(min(len(x.start), len(y.start))):
            if(np.random.uniform() < self.probability):
                tmp_start_x, tmp_slen_x = copy(x.start[i]), copy(x.slen[i])
                tmp_start_y, tmp_slen_y = copy(y.start[i]), copy(y.slen[i])
                x.start[i], y.start[i] = y.start[i], x.start[i]
                x.slen[i], y.slen[i] = y.slen[i], x.slen[i]
                if not self.validate(x):
                    x.start[i], x.slen[i] = copy(tmp_start_x), copy(tmp_slen_x)
                if not self.validate(y):
                    y.start[i], y.slen[i] = copy(tmp_start_y), copy(tmp_slen_y)
        return 0

    def write_fitness(self, x: 'fitness curve', filename):
        df = pd.DataFrame(x)
        df.to_csv(self.output_folder + '/' + filename + '.fitness.csv', index=False, header=False)

    def write_shapelets(self, x: individual, filename):
        out = {}
        out['start'] = [start for start in x.start]
        out['slen'] = [slen for slen in x.slen]
        out['cluster'] = [cluster for cluster in x.cluster] if x.cluster is not None else [-2] * len(x.start)

        df = pd.DataFrame(out, columns=['start', 'slen', 'cluster'])
        df.sort_values('cluster', inplace=True)  # unordered indizes .reset_index(inplace=True, drop=True)
        df.to_csv(self.output_folder + '/' + filename + '.shapelets.csv', index=False)
        return 0

    def make_filename(self, popsize, sigma, t_max, notes):
        filename = os.path.splitext(os.path.basename(self.ts_path))[0]  # get name without path and extension
        motifs = str(self.nsegments) + 'x' + str(self.min_support) + 'motifs'
        window_length = str(self.smin) + '-' + str(self.smax) + 'window'
        hyperparameter = str(popsize) + '_' + str(sigma) + '_' + str(t_max / 60) + '_' + str(notes)
        return 'genshapelet_' + filename + '_' + motifs + '_' + window_length + '_' + hyperparameter

    def make_individual(self):
        x = individual()
        for i in range(self.nsegments):
            x.slen.append(np.random.randint(self.smin, self.smax + 1))
            x.start.append(np.random.randint(0, len(self.ts) - x.slen[i]))

            valid = False
            attempts = 5  # this is random, right; but the whole should stay random so .. ¯\_(ツ)_/¯
            while(not valid and attempts > 0):
                valid = True
                for j in range(i):
                    if((x.start[i] + x.slen[i] <= x.start[j]) or (x.start[j] + x.slen[j] <= x.start[i])):
                        continue
                    else:
                        valid = False
                        attempts -= 1
                        x.start[i] = np.random.randint(0, len(self.ts) - x.slen[i])
                        break
            if (attempts == 0):
                # print('The individual isn\'t complete. Check nsegments and smax parameter.')
                x.slen.pop()
                x.start.pop()
                break
        return x

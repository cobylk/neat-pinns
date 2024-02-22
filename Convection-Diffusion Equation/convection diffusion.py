import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
from jax import grad, jit
from jax import lax
from matplotlib import rc
import neat


@jit
def evaluate(x: float, adj_matrix, biases, activations, aggregations):
    def relu(x):
        return jnp.maximum(0.0, x)

    def sigmoid(x):
        return 1.0 / (1.0 + jnp.exp(-x))

    def tanh(x):
        return jnp.tanh(x)

    def gauss(x):
        # -5 multiplier and clipping are there to match the python-neat library's gauss function
        return jnp.exp(-5.0 * (jnp.clip(x, -3.4, 3.4)) ** 2.0)

    def exp(x):
        return jnp.exp(x)

    def sin(x):
        return jnp.sin(x)

    def identity(x):
        return x

    def sum(x):
        return jnp.sum(x)

    def product(x):
        return jnp.prod(x)

    def aggregation(x, key):
        # key of 0 → sum; key of 1 → product
        key = jnp.astype(key, int)
        agg_list = [sum, product]
        return lax.switch(key, agg_list, x)

    def activation(x, key):
        # key of 0 → relu; 1 → sigmoid; 2 → tanh; etc.
        key = jnp.astype(key, int)
        func_list = [relu, sigmoid, tanh, gauss, exp, sin, identity]
        return lax.switch(key, func_list, x)

    node_values = jnp.zeros(biases.size)
    node_values = node_values.at[0].set(x)
    for i, func in enumerate(activations[1:], 1):
        value = 0
        value += aggregation(adj_matrix[:i, i] * node_values[:i], aggregations[i])
        value += biases[i]
        value = activation(value, func)
        node_values = node_values.at[i].set(value)

    return node_values[-1]


def eval_genome(genomes, config):
    best_fitness = -jnp.inf
    for genome_id, genome in genomes:
        a = get_graph_repr(genome)
        b = get_fitness(*a)
        if str(b) == "nan":
            genome.fitness = -jnp.inf
        else:
            genome.fitness = float(b)
        best_fitness = max(best_fitness, genome.fitness)
    print(f"Best fitness: {best_fitness}")


u = jit(evaluate)           
u_x = jit(grad(evaluate, 0))
u_xx = jit(grad(u_x, 0))


@jit
def f(x, adj_mat, bs, ac, ag):
    # return the PDE residuals at the given point
    argus = (x, adj_mat, bs, ac, ag)
    return (6 * u_x(*argus) - u_xx(*argus)) ** 2


@jit
def get_fitness(adj_mat, bs, ac, ag):
    x = jnp.linspace(0, 1, 100)


    # bfun 1 & 2 are for jax fori_loop
    # using normal for loops increases jit compilation time to upwards of an hour for 10,000 collocation points
    argus = (adj_mat, bs, ac, ag)
    def bfun2(i, val):
        return val + (f(i / 500, *argus))

    pde_loss = lax.fori_loop(0, 500, bfun2, 0.)
    pde_loss /= 500
    
    bi_loss = (u(0, *argus) - 0) ** 2 + (u(1, *argus) - 1) ** 2
    bi_loss /= 2
    return -1 * (pde_loss + bi_loss)


def filter_conns(conns, criterion, c_index):
    return {key: value for key, value in conns.items() if key[c_index] != criterion}


def topological_sort(genome):
    filtered_connections = {}
    L = []
    for key, connection in zip(list(genome.connections.keys()), list(genome.connections.values())):
        if connection.enabled:
            filtered_connections[key] = connection.weight
    running_connections = filtered_connections
    edge_endpoints = [key[1] for key in running_connections.keys()]
    edge_starts = [key[0] for key in running_connections.keys()]
    nodes = list(genome.nodes.keys())
    nodes = [node for node in nodes if node in edge_endpoints or node in edge_starts]
    nodes = [-1.] + nodes
    S = [node for node in nodes if node not in edge_endpoints]
    while S:
        n = S[0]
        nodes.remove(n)
        L.append(S.pop(0))
        running_connections = filter_conns(running_connections, n, 0)
        edge_endpoints = [key[1] for key in running_connections.keys()]
        S = [node for node in nodes if node not in edge_endpoints]

    return L


def get_biases(sorted_nodes, genome):
    return jnp.array([0.] + [genome.nodes[node].bias for node in sorted_nodes[2:]])


def get_activations(sorted_nodes, genome):
    act_dict = {'relu': 0., 'sigmoid': 1., 'tanh': 2., 'gauss': 3., 'exp': 4., 'sin': 5., 'identity': 6.}
    return jnp.array([6.] + [act_dict[genome.nodes[node].activation] for node in sorted_nodes[2:]])


def get_aggregations(sorted_nodes, genome):
    act_dict = {'sum': 0., 'product': 1.}
    return jnp.array([0.] + [act_dict[genome.nodes[node].aggregation] for node in sorted_nodes[2:]])


def get_adjacency_matrix(sorted_nodes, genome):
    matrix = jnp.zeros((len(sorted_nodes), len(sorted_nodes)))
    for key, value in genome.connections.items():
        if value.enabled:
            new_start_index = sorted_nodes.index(key[0])
            new_end_index = sorted_nodes.index(key[1])
            matrix = matrix.at[new_start_index, new_end_index].set(value.weight)
    return matrix


def get_graph_repr(genome):
    sor = topological_sort(genome)
    matr = get_adjacency_matrix(sor, genome)
    bs = get_biases(sor, genome)
    ac = get_activations(sor, genome)
    ag = get_aggregations(sor, genome)
    return matr, bs, ac, ag


def get_config():
    config_path = 'config-cd'
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation,
                       config_path)


def run(num_generations=100, checkpoint=0):
    config = get_config()
    if checkpoint == 0:
        p = neat.Population(config)
    else:
        p = neat.Checkpointer.restore_checkpoint(f"neat-checkpoint-{checkpoint}")
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50))

    return p.run(eval_genome, num_generations)


if __name__ == '__main__':

    genome = run(num_generations=1000, checkpoint=0)
    
    d = adj_mat, biases, acts, ags = get_graph_repr(genome)

    ######################
    ####  plotting!  #####
    ######################
    rc('font', **{'family': 'serif', 'serif': ['Georgia'], 'size': 12})
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.8))
    x = jnp.linspace(0, 1, 500)
    z = jnp.array([evaluate(i, *d) for i in x])
    arr = jnp.array([jnp.exp(6*((i) - 1)) for i in x])

    print(f"L2 error: {jnp.linalg.norm(arr - z)/jnp.linalg.norm(arr)}")

    axs[0].plot(x, arr, color='blue', linewidth=2, label='Reference solution')
    axs[0].plot(x, z, color='red', linewidth=2, linestyle='dashed', label='NEAT-evolved PINN solution')
    axs[0].legend()
    axs[0].set_title('Solution')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('u')

    axs[1].plot(x, jnp.abs(arr - z), color='red', linewidth=2, label='Absolute Error')
    axs[1].set_title('Absolute Error')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Error')

    axs[2].plot(x, jnp.array([f(i, *d) for i in x]), color='red', linewidth=2, label='PDE Residuals')
    axs[2].set_title('PDE Residuals')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Residuals')
    plt.tight_layout()
    plt.savefig('plots-cd.png', dpi=600)
    plt.show()


    print('\nBest genome:\n{!s}'.format(genome))

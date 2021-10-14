from tensorgp.engine import *

# Fitness function to calculate RMSE from target (Pagie Polynomial)
def calc_fit(**kwargs):
    # read parameters
    population = kwargs.get('population')
    generation = kwargs.get('generation')
    tensors = kwargs.get('tensors')
    f_path = kwargs.get('f_path')
    _stf = kwargs.get('stf')
    target = kwargs.get('target')
    
    print(np.array(tensors).shape)
    
    fn = f_path + "gen_" + str(generation).zfill(5)
    fitness = []
    times = []
    best_ind = 0

    # set objective function according to min/max
    fit = 0
    condition = lambda: (fit < max_fit)  # minimizing
    max_fit = float('inf')

    for i in range(len(tensors)):

        start_ind = time.time()
        fit = tf_rmse(tensors[i], target).numpy()

        if condition():
            max_fit = fit
            best_ind = i

        times.append((time.time() - start_ind) * 1000.0)
        fitness.append(fit)
        population[i]['fitness'] = fit

    #if generation == gens:
    #    save_image(tensors[best_ind], best_ind, fn, 2)

    return population, population[best_ind]


# Different types of function sets
extended_fset = {'max', 'min', 'abs', 'add', 'and', 'or', 'mult', 'sub', 'xor', 'neg', 'cos', 'sin', 'tan', 'sqrt', 'div', 'exp', 'log', 'warp'}
simple_set = {'add', 'sub', 'mult', 'div', 'sin', 'tan', 'cos'}
normal_set = {'add', 'mult', 'sub', 'div', 'cos', 'sin', 'tan', 'abs', 'sign', 'pow'}


if __name__ == "__main__":

    # GP params
    dev = '/gpu:0'  # device to run, write '/cpu_0' to tun on cpu
    gens = 49  # 50
    pop_size = 50  # 50
    tour_size = 3
    mut_rate = 0.1
    cross_rate = 0.9
    max_tree_dep = 10
    max_init_depth = 10
    elite_size = 1 # 0 to turn off
    runs = 1 # Number of average runs

    # problems
    pagie = "add(div(scalar(1.0), add(scalar(1.0), div(scalar(1.0), mult(mult(x, x), mult(x, x))))), div(scalar(1.0), add(scalar(1.0), div(scalar(1.0), mult(mult(y, y), mult(y, y))))))"
    keijzer11 = "add(mult(x, y), sin(mult(sub(x, scalar(1.0), sub(y, scalar(1.0)))))"
    korns3 = "add(scalar(-5.41), mult(scalar(4.9), div(sub(v, add(x, div(y, w))), mult(scalar(3.0, w)))))"

    problems = [pagie]  # Add to run more problems

    # Domains dimensions
    test_cases = [[64, 64]]

    for p in problems:

        for res in test_cases:

            for r in range(runs):

                #seeds = random.randint(0, 0x7fffffff)
                seeds = 39485793482 # reproducibility

                # create engine
                engine = Engine(fitness_func=calc_fit,
                                population_size=pop_size,
                                tournament_size=tour_size,
                                mutation_rate=mut_rate,
                                crossover_rate=cross_rate,
                                max_tree_depth=max_tree_dep,
                                target_dims=res,
                                target=pagie,
                                #elitism=elite_size,
                                method='ramped half-and-half',
                                max_init_depth=max_init_depth,
                                objective='minimizing',
                                device=dev,
                                stop_criteria='generation',
                                stop_value=gens,
                                effective_dims=2,
                                min_domain=-5,
                                max_domain=5,
                                operators=normal_set,
                                seed=seeds,
                                save_to_file=10,
                                save_graphics=False,
                                show_graphics=False,
                                write_log=False,
                                write_gen_stats=False,
                                read_init_pop_from_file=None)

                # run evolutionary process
                engine.run()
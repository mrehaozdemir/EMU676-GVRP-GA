# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %% Imports and setup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from copy import deepcopy

# Reproducibility için sabit seed
random.seed(42)
np.random.seed(42)

print("Imports OK")

# %% Toy GVRP instance (5 customers + depot)
# Node 0 = depot, nodes 1-5 = customers
coords = np.array([
    [50, 50],  # 0: depot
    [20, 30],  # 1: customer
    [80, 70],  # 2: customer
    [30, 80],  # 3: customer
    [70, 20],  # 4: customer
    [60, 60],  # 5: customer
])

demands = np.array([0, 10, 15, 8, 12, 20])  # depot demand = 0
Q = 50    # vehicle capacity
m = 3     # max number of vehicles
n = len(coords) - 1  # number of customers

# Distance matrix (Euclidean)
n_nodes = len(coords)
dist = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    for j in range(n_nodes):
        dist[i, j] = np.sqrt(
            (coords[i, 0] - coords[j, 0])**2 +
            (coords[i, 1] - coords[j, 1])**2
        )

# Xiao et al. (2012) fuel consumption parameters
rho_0 = 0.165       # empty vehicle FCR (L/km)
rho_star = 0.377    # fully loaded vehicle FCR (L/km)
alpha = (rho_star - rho_0) / Q  # load-dependent slope
F = 50              # fixed cost per vehicle used
c_0 = 1.0           # unit fuel cost

print(f"Instance: {n} customers, capacity={Q}, max vehicles={m}")
print(f"Distance matrix shape: {dist.shape}")
print(f"Alpha (load slope): {alpha:.5f}")

# %% Decoder: permutation -> vehicle routes
def decode_permutation(perm, demands, Q):
    """
    Müşteri permütasyonunu araç rotalarına çevirir.
    Soldan sağa tarar, kapasite dolunca yeni araç açar.
    Her rota depodan başlar ve depoda biter.
    """
    routes = []
    current_route = [0]   # depot ile başla
    current_load = 0
    
    for customer in perm:
        if current_load + demands[customer] <= Q:
            current_route.append(customer)
            current_load += demands[customer]
        else:
            # Kapasite yetmedi, mevcut rotayı kapat, yeni araç aç
            current_route.append(0)
            routes.append(current_route)
            current_route = [0, customer]
            current_load = demands[customer]
    
    current_route.append(0)  # son rotayı depoda kapat
    routes.append(current_route)
    return routes


# Test edelim
test_perm = [3, 1, 5, 2, 4]
test_routes = decode_permutation(test_perm, demands, Q)
print("Test permutation:", test_perm)
print("Decoded routes:")
for k, r in enumerate(test_routes):
    print(f"  Vehicle {k+1}: {r}")


# %% Fitness function: total cost (fixed + fuel)
def calculate_fuel_cost(routes, dist, demands, rho_0, alpha, c_0, F):
    """
    Xiao et al. (2012) objective:
      H = sum(F * vehicles_used) 
        + sum(c_0 * d_ij * (rho_0 + alpha * y_ij)) for all arcs
    
    y_ij = arc (i,j) üzerinde taşınan yük.
    Depodan çıkarken aracın toplam yükü, her müşteriden sonra azalır.
    """
    total_cost = 0.0
    
    for route in routes:
        if len(route) <= 2:  # boş rota (sadece depot-depot)
            continue
        
        # Bu aracın taşıdığı toplam talep
        route_load = sum(demands[c] for c in route if c != 0)
        
        total_cost += F  # araç kullanıldı, fixed cost ekle
        
        current_load = route_load  # depodan çıkarken full
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            d = dist[from_node, to_node]
            
            # Yakıt: c_0 * d * (rho_0 + alpha * current_load)
            fuel_arc = c_0 * d * (rho_0 + alpha * current_load)
            total_cost += fuel_arc
            
            # Müşteriye teslimat yapıldıysa yük azalır
            if to_node != 0:
                current_load -= demands[to_node]
    
    return total_cost


# Test
test_cost = calculate_fuel_cost(test_routes, dist, demands, rho_0, alpha, c_0, F)
print(f"Test fuel cost: {test_cost:.2f}")

# %% Initial population (random + greedy mix)
def greedy_nearest_neighbor(customers, dist):
    """En yakın komşu sezgisel ile bir permütasyon üretir."""
    unvisited = customers.copy()
    perm = []
    current = 0  # depot
    while unvisited:
        next_c = min(unvisited, key=lambda c: dist[current, c])
        perm.append(next_c)
        unvisited.remove(next_c)
        current = next_c
    return perm


def initial_population(n, pop_size, dist, greedy_ratio=0.2):
    """
    Başlangıç popülasyonu: %80 random, %20 greedy (NN sezgiseli + ufak karıştırma).
    Greedy bireyler convergence'ı hızlandırır ama çeşitliliği bozmasın diye az tutulur.
    """
    customers = list(range(1, n + 1))
    population = []
    n_random = int(pop_size * (1 - greedy_ratio))
    
    for _ in range(n_random):
        perm = customers.copy()
        random.shuffle(perm)
        population.append(perm)
    
    for _ in range(pop_size - n_random):
        perm = greedy_nearest_neighbor(customers, dist)
        # çeşitlilik için birkaç swap
        for _ in range(max(1, len(perm) // 5)):
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        population.append(perm)
    
    return population


# Test
pop = initial_population(n, pop_size=10, dist=dist)
print(f"Population size: {len(pop)}")
print(f"First individual:  {pop[0]}")
print(f"Last (greedy):     {pop[-1]}")


# %% Tournament selection
def tournament_selection(population, fitnesses, k=3):
    """
    k tane rastgele birey seç, en iyi (en düşük fitness) olanı döndür.
    k büyüdükçe selection pressure artar; k=3 standart ve dengeli.
    """
    indices = random.sample(range(len(population)), k)
    best_idx = min(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


# %% Order Crossover (OX)
def order_crossover(parent1, parent2):
    """
    OX: parent1'den bir segment kopyala, parent2'nin sırasını koruyarak boşlukları doldur.
    Permütasyon yapısını koruduğu için VRP'ye uygundur.
    """
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    
    child = [None] * size
    child[a:b+1] = parent1[a:b+1]
    
    used = set(child[a:b+1])
    p2_filtered = [c for c in parent2 if c not in used]
    
    pos = 0
    for i in range(size):
        if child[i] is None:
            child[i] = p2_filtered[pos]
            pos += 1
    return child


# %% Mutation operators
def swap_mutation(perm):
    """İki rastgele müşteriyi yer değiştirir."""
    new_perm = perm.copy()
    i, j = random.sample(range(len(new_perm)), 2)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return new_perm


def inversion_mutation(perm):
    """Rastgele bir segmenti ters çevirir (2-opt benzeri etki)."""
    new_perm = perm.copy()
    a, b = sorted(random.sample(range(len(new_perm)), 2))
    new_perm[a:b+1] = reversed(new_perm[a:b+1])
    return new_perm


def mutate(perm, pm):
    """pm olasılıkla swap veya inversion uygular (50/50)."""
    if random.random() < pm:
        if random.random() < 0.5:
            return swap_mutation(perm)
        else:
            return inversion_mutation(perm)
    return perm.copy()


# Test
p1 = [1, 2, 3, 4, 5]
p2 = [5, 4, 3, 2, 1]
child = order_crossover(p1, p2)
print(f"Parent1: {p1}")
print(f"Parent2: {p2}")
print(f"OX child: {child}")
print(f"Swap mut: {swap_mutation(p1)}")
print(f"Inv mut:  {inversion_mutation(p1)}")

# %% Main GA loop
def run_ga(n, demands, Q, dist, rho_0, alpha, c_0, F,
           pop_size=100, n_gen=200, p_crossover=0.9, p_mutation=0.1,
           tournament_k=3, elite_size=2, no_improve_limit=50, verbose=False):
    """
    Ana GA döngüsü.
    - Elitism: her jenerasyonda en iyi `elite_size` birey doğrudan aktarılır
      (en iyi çözümün kaybolmasını önler)
    - Stopping: ya max generation ya da `no_improve_limit` jenerasyon
      boyunca iyileşme olmazsa erken durur
    """
    # Başlangıç popülasyonu ve fitness değerleri
    population = initial_population(n, pop_size, dist)
    fitnesses = []
    for perm in population:
        routes = decode_permutation(perm, demands, Q)
        cost = calculate_fuel_cost(routes, dist, demands, rho_0, alpha, c_0, F)
        fitnesses.append(cost)
    
    best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
    best_perm = population[best_idx].copy()
    best_cost = fitnesses[best_idx]
    
    history = [best_cost]
    no_improve_count = 0
    
    for gen in range(n_gen):
        # Elitism: en iyi bireyleri aktar
        sorted_idx = sorted(range(pop_size), key=lambda i: fitnesses[i])
        new_population = [population[i].copy() for i in sorted_idx[:elite_size]]
        
        # Geri kalan popülasyonu üret
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, k=tournament_k)
            parent2 = tournament_selection(population, fitnesses, k=tournament_k)
            
            if random.random() < p_crossover:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            child = mutate(child, p_mutation)
            new_population.append(child)
        
        # Yeni popülasyon fitness'ları
        population = new_population
        fitnesses = []
        for perm in population:
            routes = decode_permutation(perm, demands, Q)
            cost = calculate_fuel_cost(routes, dist, demands, rho_0, alpha, c_0, F)
            fitnesses.append(cost)
        
        # Best'i güncelle
        gen_best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
        gen_best_cost = fitnesses[gen_best_idx]
        
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_perm = population[gen_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        history.append(best_cost)
        
        if verbose and (gen + 1) % 20 == 0:
            print(f"  Gen {gen+1}: best={best_cost:.2f}")
        
        # Erken durma kontrolü
        if no_improve_count >= no_improve_limit:
            if verbose:
                print(f"  Early stop at gen {gen+1} (no improvement for {no_improve_limit} gens)")
            break
    
    return best_perm, best_cost, history


# Toy instance üzerinde test
print("Running GA on toy instance...")
best_perm, best_cost, history = run_ga(
    n, demands, Q, dist, rho_0, alpha, c_0, F,
    pop_size=50, n_gen=100, verbose=True
)
print(f"\nBest permutation: {best_perm}")
print(f"Best cost: {best_cost:.2f}")
print("Best routes:")
for k, r in enumerate(decode_permutation(best_perm, demands, Q)):
    print(f"  Vehicle {k+1}: {r}")
    
    # %% Instance generator (synthetic GVRP instances)
def generate_instance(n_customers, capacity, seed=None,
                      coord_range=(0, 100), demand_range=(1, 30)):
    """
    Sentetik bir GVRP instance üretir.
    - Koordinatlar uniform random (Christofides tarzı)
    - Talepler küçük tamsayılar (CVRPLIB konvansiyonu)
    - Depo orta noktada
    """
    rng = np.random.RandomState(seed)
    
    depot_coord = np.array([(coord_range[0] + coord_range[1]) / 2] * 2)
    customer_coords = rng.uniform(coord_range[0], coord_range[1], (n_customers, 2))
    coords = np.vstack([depot_coord.reshape(1, 2), customer_coords])
    
    demands = np.zeros(n_customers + 1)
    demands[1:] = rng.randint(demand_range[0], demand_range[1] + 1, n_customers)
    
    n_nodes = n_customers + 1
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist[i, j] = np.sqrt(
                (coords[i, 0] - coords[j, 0])**2 +
                (coords[i, 1] - coords[j, 1])**2
            )
    
    # Yaklaşık ihtiyaç duyulan araç sayısı + güvenlik payı
    m_est = int(np.ceil(demands.sum() / capacity)) + 2
    
    return {
        'coords': coords,
        'demands': demands,
        'dist': dist,
        'Q': capacity,
        'n': n_customers,
        'm': m_est
    }


# %% Create 10 benchmark instances (small, medium, large mix)
INSTANCE_SPECS = [
    # (name, n_customers, capacity)
    ('S1_n10',   10,  50),
    ('S2_n15',   15,  50),
    ('S3_n20',   20,  60),
    ('M1_n30',   30,  80),
    ('M2_n40',   40, 100),
    ('M3_n50',   50, 100),
    ('L1_n75',   75, 120),
    ('L2_n100', 100, 150),
    ('L3_n125', 125, 150),
    ('L4_n150', 150, 200),
]

instances = {}
print(f"{'Name':<10} {'n':>4} {'Q':>4} {'TotalDemand':>12} {'~Vehicles':>10}")
print("-" * 50)
for i, (name, n_c, cap) in enumerate(INSTANCE_SPECS):
    inst = generate_instance(n_c, cap, seed=100 + i)
    instances[name] = inst
    print(f"{name:<10} {n_c:>4} {cap:>4} {int(inst['demands'].sum()):>12} {inst['m']:>10}")


# %% GA wrapper that takes instance dict
def run_ga_on_instance(inst, pop_size=100, n_gen=200, p_mutation=0.1,
                       seed=None, verbose=False):
    """run_ga'yı bir instance dict üzerinde çalıştıran sarmalayıcı."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    return run_ga(
        n=inst['n'],
        demands=inst['demands'],
        Q=inst['Q'],
        dist=inst['dist'],
        rho_0=rho_0,
        alpha=(rho_star - rho_0) / inst['Q'],  # alpha instance'a özgü
        c_0=c_0,
        F=F,
        pop_size=pop_size,
        n_gen=n_gen,
        p_mutation=p_mutation,
        verbose=verbose
    )


# %% Quick sanity check: 1 run on small instance
print("\nQuick test on S1_n10...")
import time
t0 = time.time()
best_perm, best_cost, history = run_ga_on_instance(
    instances['S1_n10'], pop_size=50, n_gen=100, seed=42, verbose=True
)
elapsed = time.time() - t0
print(f"\nBest cost: {best_cost:.2f}")
print(f"Elapsed: {elapsed:.2f} s")
print(f"Generations run: {len(history) - 1}")

# %% Batch runner: tüm instance'larda çoklu run + istatistik
def run_batch(instances, n_runs=5, pop_size=100, n_gen=200, p_mutation=0.15):
    """
    Her instance'da n_runs kez GA çalıştırır.
    Her run farklı seed kullanır -> istatistiksel olarak anlamlı sonuç.
    """
    results = []
    histories = {}
    
    for name, inst in instances.items():
        print(f"Running {name} (n={inst['n']})...")
        
        run_costs = []
        run_times = []
        run_histories = []
        
        for run_id in range(n_runs):
            seed = 1000 + run_id
            t0 = time.time()
            _, best_cost, history = run_ga_on_instance(
                inst, pop_size=pop_size, n_gen=n_gen,
                p_mutation=p_mutation, seed=seed, verbose=False
            )
            elapsed = time.time() - t0
            
            run_costs.append(best_cost)
            run_times.append(elapsed)
            run_histories.append(history)
        
        costs = np.array(run_costs)
        times = np.array(run_times)
        
        result = {
            'instance': name,
            'n': inst['n'],
            'Q': inst['Q'],
            'best': round(costs.min(), 2),
            'mean': round(costs.mean(), 2),
            'std': round(costs.std(), 2),
            'worst': round(costs.max(), 2),
            'cv_pct': round(costs.std() / costs.mean() * 100, 2),  # variation %
            'avg_time_s': round(times.mean(), 3),
            'n_runs': n_runs
        }
        results.append(result)
        histories[name] = run_histories
        
        print(f"  Best={result['best']:.2f}  Mean={result['mean']:.2f}  "
              f"Std={result['std']:.2f}  CV={result['cv_pct']:.2f}%  "
              f"Time={result['avg_time_s']:.2f}s")
    
    return pd.DataFrame(results), histories


# %% Run batch experiments
print("=" * 60)
print("BATCH EXPERIMENTS: 5 runs per instance")
print("=" * 60)

results_df, histories = run_batch(
    instances, n_runs=5, pop_size=100, n_gen=200, p_mutation=0.15
)

print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))

# CSV'ye kaydet
results_df.to_csv('gvrp_ga_results.csv', index=False)
print("\nSaved to gvrp_ga_results.csv")

# %% Convergence plots (10 instances)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for idx, (name, run_hists) in enumerate(histories.items()):
    ax = axes[idx]
    max_len = max(len(h) for h in run_hists)
    padded = np.full((len(run_hists), max_len), np.nan)
    for i, h in enumerate(run_hists):
        padded[i, :len(h)] = h
        padded[i, len(h):] = h[-1]  # erken durmada son değerle doldur
    
    for i in range(len(run_hists)):
        ax.plot(padded[i], alpha=0.3, color='gray', linewidth=0.8)
    
    mean_curve = np.nanmean(padded, axis=0)
    ax.plot(mean_curve, color='C0', linewidth=2, label='Mean')
    
    ax.set_title(f"{name} (n={instances[name]['n']})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fuel cost")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle("GA Convergence Behavior Across Instances (5 runs each)", fontsize=14)
plt.tight_layout()
plt.savefig('convergence_plots.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved convergence_plots.png")


# %% Toy example illustration (sequence matters)
def plot_route(ax, coords, routes, demands, title, fuel_cost):
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    ax.scatter(coords[1:, 0], coords[1:, 1],
               s=300, c='lightblue', edgecolors='navy',
               linewidth=2, zorder=3)
    for i in range(1, len(coords)):
        label = str(i) + "\nd=" + str(int(demands[i]))
        ax.annotate(label, coords[i], ha='center', va='center',
                    fontsize=8, fontweight='bold', zorder=4)
    ax.scatter(coords[0, 0], coords[0, 1],
               s=400, c='red', marker='s', edgecolors='darkred',
               linewidth=2, zorder=3)
    ax.annotate("0", coords[0], ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=4)
    for k, r in enumerate(routes):
        if len(r) <= 2:
            continue
        color = colors[k % len(colors)]
        for i in range(len(r) - 1):
            ax.annotate("", xy=coords[r[i+1]], xytext=coords[r[i]],
                        arrowprops=dict(arrowstyle='->', color=color, lw=2),
                        zorder=2)
    full_title = title + "\nFuel cost = " + ("%.2f" % fuel_cost)
    ax.set_title(full_title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


# Toy instance üzerinde GA'yı tekrar çalıştır (best_perm batch'ten kalmış olabilir)
random.seed(42)
np.random.seed(42)
toy_best_perm, toy_best_cost, _ = run_ga(
    n, demands, Q, dist, rho_0, alpha, c_0, F,
    pop_size=50, n_gen=100, verbose=False
)
print("Toy GA best perm:", toy_best_perm)
print("Toy GA best cost:", "%.2f" % toy_best_cost)

# Karşılaştırma: naive vs GA
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

naive_perm = sorted(range(1, n + 1), key=lambda c: demands[c])
naive_routes = decode_permutation(naive_perm, demands, Q)
naive_cost = calculate_fuel_cost(naive_routes, dist, demands, rho_0, alpha, c_0, F)
naive_title = "Naive: light demands first\nPerm: " + str(naive_perm)
plot_route(axes[0], coords, naive_routes, demands, naive_title, naive_cost)

ga_routes = decode_permutation(toy_best_perm, demands, Q)
ga_cost = calculate_fuel_cost(ga_routes, dist, demands, rho_0, alpha, c_0, F)
ga_title = "GA solution\nPerm: " + str(toy_best_perm)
plot_route(axes[1], coords, ga_routes, demands, ga_title, ga_cost)

plt.suptitle("Toy GVRP: Visit Sequence Affects Fuel Consumption", fontsize=13)
plt.tight_layout()
plt.savefig('toy_illustration.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved toy_illustration.png")
print("Naive cost: " + ("%.2f" % naive_cost))
print("GA cost:    " + ("%.2f" % ga_cost))
improvement = (naive_cost - ga_cost) / naive_cost * 100
print("Improvement: " + ("%.1f" % improvement) + "%")

# %% Methodology workflow illustration (4-panel)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# --- Panel 1: Problem (instance only) ---
ax = axes[0, 0]
ax.scatter(coords[1:, 0], coords[1:, 1],
           s=400, c='lightblue', edgecolors='navy', linewidth=2, zorder=3)
for i in range(1, len(coords)):
    label = str(i) + "\nd=" + str(int(demands[i]))
    ax.annotate(label, coords[i], ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=4)
ax.scatter(coords[0, 0], coords[0, 1],
           s=500, c='red', marker='s', edgecolors='darkred', linewidth=2, zorder=3)
ax.annotate("Depot", coords[0], ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=4)
ax.set_title("(a) Problem: 5-customer GVRP instance\n"
             "Capacity Q=50, demands shown on each node", fontsize=11)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# --- Panel 2: Encoding (chromosome) ---
ax = axes[0, 1]
ax.axis('off')
example_perm = [3, 1, 5, 2, 4]
ax.set_title("(b) Encoding: chromosome as permutation", fontsize=11)
# Chromosome cells
cell_w, cell_h = 0.12, 0.15
start_x = 0.18
y_center = 0.55
for i, gene in enumerate(example_perm):
    rect = plt.Rectangle((start_x + i * cell_w, y_center),
                          cell_w, cell_h,
                          facecolor='lightyellow', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(start_x + i * cell_w + cell_w / 2, y_center + cell_h / 2,
            str(gene), ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(start_x + i * cell_w + cell_w / 2, y_center - 0.05,
            "gene " + str(i + 1), ha='center', va='center', fontsize=8)
ax.text(0.5, 0.85, "Chromosome = " + str(example_perm),
        ha='center', fontsize=12, fontweight='bold')
ax.text(0.5, 0.3,
        "Each gene = customer ID\nOrder = visit sequence",
        ha='center', va='center', fontsize=10)
ax.text(0.5, 0.1,
        "Permutation length = n (no. of customers)",
        ha='center', va='center', fontsize=9, style='italic')

# --- Panel 3: Decoding (routes from chromosome) ---
ax = axes[1, 0]
example_routes = decode_permutation(example_perm, demands, Q)
example_cost = calculate_fuel_cost(example_routes, dist, demands, rho_0, alpha, c_0, F)
colors = ['C0', 'C1', 'C2']
ax.scatter(coords[1:, 0], coords[1:, 1],
           s=400, c='lightblue', edgecolors='navy', linewidth=2, zorder=3)
for i in range(1, len(coords)):
    ax.annotate(str(i), coords[i], ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=4)
ax.scatter(coords[0, 0], coords[0, 1],
           s=500, c='red', marker='s', edgecolors='darkred', linewidth=2, zorder=3)
ax.annotate("0", coords[0], ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=4)
for k, r in enumerate(example_routes):
    if len(r) <= 2:
        continue
    color = colors[k % len(colors)]
    for i in range(len(r) - 1):
        ax.annotate("", xy=coords[r[i+1]], xytext=coords[r[i]],
                    arrowprops=dict(arrowstyle='->', color=color, lw=2),
                    zorder=2)
title_str = "(c) Decoding: split by capacity\n"
for k, r in enumerate(example_routes):
    if len(r) > 2:
        title_str += "Vehicle " + str(k+1) + ": " + str(r) + "\n"
title_str += "Fuel cost = " + ("%.2f" % example_cost)
ax.set_title(title_str, fontsize=10)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# --- Panel 4: GA best solution ---
ax = axes[1, 1]
ga_routes_show = decode_permutation(toy_best_perm, demands, Q)
ga_cost_show = calculate_fuel_cost(ga_routes_show, dist, demands, rho_0, alpha, c_0, F)
ax.scatter(coords[1:, 0], coords[1:, 1],
           s=400, c='lightgreen', edgecolors='darkgreen', linewidth=2, zorder=3)
for i in range(1, len(coords)):
    ax.annotate(str(i), coords[i], ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=4)
ax.scatter(coords[0, 0], coords[0, 1],
           s=500, c='red', marker='s', edgecolors='darkred', linewidth=2, zorder=3)
ax.annotate("0", coords[0], ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=4)
for k, r in enumerate(ga_routes_show):
    if len(r) <= 2:
        continue
    color = colors[k % len(colors)]
    for i in range(len(r) - 1):
        ax.annotate("", xy=coords[r[i+1]], xytext=coords[r[i]],
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5),
                    zorder=2)
title_str = "(d) GA best solution\n"
title_str += "Chromosome: " + str(toy_best_perm) + "\n"
title_str += "Fuel cost = " + ("%.2f" % ga_cost_show)
ax.set_title(title_str, fontsize=10)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.suptitle("GA-GVRP Methodology Overview on a 5-Customer Toy Instance",
             fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('methodology_illustration.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved methodology_illustration.png")
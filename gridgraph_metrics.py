import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import math
import random
from typing import Tuple, Optional, List
import time

def heuristic(x1: int, y1: int, x2: int, y2: int) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos.

    Args:
        x1: Coordenada x del primer punto
        y1: Coordenada y del primer punto
        x2: Coordenada x del segundo punto
        y2: Coordenada y del segundo punto

    Returns:
        float: Distancia euclidiana entre los puntos
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

class MemoryManager:
    """Gestiona la memoria para el algoritmo A* en CUDA."""

    def __init__(self, width: int, height: int, max_gpu_memory: int = 1024*1024*1024):
        """
        Inicializa el gestor de memoria.

        Args:
            width: Ancho del grid
            height: Alto del grid
            max_gpu_memory: Memoria máxima de GPU en bytes (default 1GB)
        """
        self.width = width
        self.height = height
        self.max_gpu_memory = max_gpu_memory
        self.total_size = width * height

        self.grid_size = self.total_size * np.int32().itemsize
        self.score_size = self.total_size * np.float32().itemsize
        self.bool_size = self.total_size * np.bool_().itemsize

        # Añadir espacio para contador de estados
        self.counter_size = np.int32().itemsize

        # Calcular tamaño total necesario
        self.total_memory_needed = (
            self.grid_size +  
            2 * self.score_size +  
            2 * self.grid_size + 
            2 * self.bool_size +  
            np.bool_().itemsize +
            self.counter_size  # Espacio para contador de estados
        )

        # Determinar si necesitamos particionamiento
        self.needs_partitioning = self.total_memory_needed > max_gpu_memory
        self.partition_size = self.calculate_partition_size() if self.needs_partitioning else self.total_size

    def calculate_partition_size(self) -> int:
        """Calcula el tamaño óptimo de partición basado en la memoria disponible."""
        memory_per_cell = (
            np.int32().itemsize * 3 + 
            np.float32().itemsize * 2 +  
            np.bool_().itemsize * 2 +
            np.int32().itemsize  # Incluir espacio para contador
        )
        return (self.max_gpu_memory // memory_per_cell) // 2

    def allocate_memory(self) -> Tuple[dict, dict]:
        """
        Aloca memoria en GPU y memoria zero-copy según sea necesario.

        Returns:
            Tuple[dict, dict]: Diccionarios con buffers GPU y zero-copy
        """
        gpu_buffers = {}
        zero_copy_buffers = {}

        if self.needs_partitioning:
            zero_copy_buffers.update({
                'grid': cuda.pagelocked_empty((self.height, self.width), np.int32),
                'g_score': cuda.pagelocked_empty(self.total_size, np.float32),
                'f_score': cuda.pagelocked_empty(self.total_size, np.float32)
            })

            gpu_buffers.update({
                'partition_grid': cuda.mem_alloc(self.partition_size * np.int32().itemsize),
                'partition_g_score': cuda.mem_alloc(self.partition_size * np.float32().itemsize),
                'partition_f_score': cuda.mem_alloc(self.partition_size * np.float32().itemsize)
            })
        else:
            gpu_buffers.update({
                'grid': cuda.mem_alloc(self.grid_size),
                'g_score': cuda.mem_alloc(self.score_size),
                'f_score': cuda.mem_alloc(self.score_size)
            })

        gpu_buffers.update({
            'came_from_x': cuda.mem_alloc(self.grid_size),
            'came_from_y': cuda.mem_alloc(self.grid_size),
            'open_set': cuda.mem_alloc(self.bool_size),
            'closed_set': cuda.mem_alloc(self.bool_size),
            'found': cuda.mem_alloc(np.bool_().itemsize),
            'expanded_states': cuda.mem_alloc(np.int32().itemsize)  # Contador de estados expandidos
        })

        return gpu_buffers, zero_copy_buffers

class AStarCUDA:
    def __init__(self, grid: np.ndarray, max_gpu_memory: int = 1024*1024*1024):
        """
        Inicializa el algoritmo A* en CUDA.

        Args:
            grid: Grid numpy con obstáculos
            max_gpu_memory: Memoria máxima de GPU en bytes (default 1GB)
        """
        self.grid = grid.astype(np.int32)
        self.height, self.width = grid.shape
        self.heuristic = heuristic
        self.expanded_states = 0  # Contador de estados expandidos

        self.memory_manager = MemoryManager(self.width, self.height, max_gpu_memory)
        self.gpu_buffers, self.zero_copy_buffers = self.memory_manager.allocate_memory()

        self.mod = SourceModule("""
        __device__ float heuristic(int x1, int y1, int x2, int y2) {
            return sqrtf(powf(x2 - x1, 2.0f) + powf(y2 - y1, 2.0f));
        }

        __global__ void astar_kernel(
            const int *grid,
            const int partition_start,
            const int partition_size,
            const int width,
            const int height,
            const int start_x,
            const int start_y,
            const int goal_x,
            const int goal_y,
            float *g_score,
            float *f_score,
            int *came_from_x,
            int *came_from_y,
            bool *open_set,
            bool *closed_set,
            bool *found,
            int *expanded_states
        ) {
            int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (local_idx >= partition_size) return;

            int idx = partition_start + local_idx;
            if (idx >= width * height) return;

            if (!open_set[idx] || closed_set[idx]) return;

            // Incrementar contador de estados expandidos
            atomicAdd(expanded_states, 1);

            int x = idx % width;
            int y = idx / width;

            if (x == goal_x && y == goal_y) {
                *found = true;
                return;
            }

            int dx[8] = {-1, -1, -1,  0,  0,  1, 1, 1};
            int dy[8] = {-1,  0,  1, -1,  1, -1, 0, 1};
            float costs[8] = {1.414f, 1.0f, 1.414f, 1.0f, 1.0f, 1.414f, 1.0f, 1.414f};

            for (int i = 0; i < 8; i++) {
                int new_x = x + dx[i];
                int new_y = y + dy[i];

                if (new_x < 0 || new_x >= width || new_y < 0 || new_y >= height) continue;

                int new_idx = new_y * width + new_x;
                if (grid[new_idx] == 1) continue;

                float tentative_g = g_score[idx] + costs[i];

                if (tentative_g < g_score[new_idx]) {
                    came_from_x[new_idx] = x;
                    came_from_y[new_idx] = y;
                    g_score[new_idx] = tentative_g;
                    f_score[new_idx] = tentative_g + heuristic(new_x, new_y, goal_x, goal_y);
                    open_set[new_idx] = true;
                }
            }

            open_set[idx] = false;
            closed_set[idx] = true;
        }
        """)

        self.astar_kernel = self.mod.get_function("astar_kernel")

    def process_partition(self, partition_start: int, partition_size: int,
                        start: Tuple[int, int], goal: Tuple[int, int]) -> None:
        """Procesa una partición del grid."""
        start_x, start_y = start
        goal_x, goal_y = goal

        block_size = 256
        grid_size = (partition_size + block_size - 1) // block_size

        if self.memory_manager.needs_partitioning:
            partition_slice = slice(partition_start, partition_start + partition_size)
            cuda.memcpy_htod(self.gpu_buffers['partition_grid'],
                           self.zero_copy_buffers['grid'].ravel()[partition_slice])
            cuda.memcpy_htod(self.gpu_buffers['partition_g_score'],
                           self.zero_copy_buffers['g_score'][partition_slice])
            cuda.memcpy_htod(self.gpu_buffers['partition_f_score'],
                           self.zero_copy_buffers['f_score'][partition_slice])

            grid_buffer = self.gpu_buffers['partition_grid']
            g_score_buffer = self.gpu_buffers['partition_g_score']
            f_score_buffer = self.gpu_buffers['partition_f_score']
        else:
            grid_buffer = self.gpu_buffers['grid']
            g_score_buffer = self.gpu_buffers['g_score']
            f_score_buffer = self.gpu_buffers['f_score']

        self.astar_kernel(
            grid_buffer,
            np.int32(partition_start),
            np.int32(partition_size),
            np.int32(self.width),
            np.int32(self.height),
            np.int32(start_x),
            np.int32(start_y),
            np.int32(goal_x),
            np.int32(goal_y),
            g_score_buffer,
            f_score_buffer,
            self.gpu_buffers['came_from_x'],
            self.gpu_buffers['came_from_y'],
            self.gpu_buffers['open_set'],
            self.gpu_buffers['closed_set'],
            self.gpu_buffers['found'],
            self.gpu_buffers['expanded_states'],
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )

        if self.memory_manager.needs_partitioning:
            cuda.memcpy_dtoh(self.zero_copy_buffers['g_score'][partition_slice],
                           self.gpu_buffers['partition_g_score'])
            cuda.memcpy_dtoh(self.zero_copy_buffers['f_score'][partition_slice],
                           self.gpu_buffers['partition_f_score'])

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Encuentra el camino más corto entre dos puntos usando A*.

        Args:
            start: Tupla (x, y) del punto inicial
            goal: Tupla (x, y) del punto final

        Returns:
            Optional[List[Tuple[int, int]]]: Lista de coordenadas del camino o None si no hay camino
        """
        # Inicializar contador de estados expandidos
        expanded_states = np.zeros(1, dtype=np.int32)
        cuda.memcpy_htod(self.gpu_buffers['expanded_states'], expanded_states)
        
        if self.memory_manager.needs_partitioning:
            self.zero_copy_buffers['g_score'].fill(np.inf)
            self.zero_copy_buffers['f_score'].fill(np.inf)
            start_idx = start[1] * self.width + start[0]
            self.zero_copy_buffers['g_score'][start_idx] = 0
            self.zero_copy_buffers['f_score'][start_idx] = self.heuristic(*start, *goal)
            cuda.memcpy_htod(self.gpu_buffers['grid'], self.grid)
        else:
            g_score = np.full(self.memory_manager.total_size, np.inf, dtype=np.float32)
            f_score = np.full(self.memory_manager.total_size, np.inf, dtype=np.float32)
            start_idx = start[1] * self.width + start[0]
            g_score[start_idx] = 0
            f_score[start_idx] = self.heuristic(*start, *goal)
            cuda.memcpy_htod(self.gpu_buffers['grid'], self.grid)
            cuda.memcpy_htod(self.gpu_buffers['g_score'], g_score)
            cuda.memcpy_htod(self.gpu_buffers['f_score'], f_score)

        open_set = np.zeros(self.memory_manager.total_size, dtype=np.bool_)
        closed_set = np.zeros(self.memory_manager.total_size, dtype=np.bool_)
        came_from_x = np.full(self.memory_manager.total_size, -1, dtype=np.int32)
        came_from_y = np.full(self.memory_manager.total_size, -1, dtype=np.int32)

        open_set[start_idx] = True

        cuda.memcpy_htod(self.gpu_buffers['open_set'], open_set)
        cuda.memcpy_htod(self.gpu_buffers['closed_set'], closed_set)
        cuda.memcpy_htod(self.gpu_buffers['came_from_x'], came_from_x)
        cuda.memcpy_htod(self.gpu_buffers['came_from_y'], came_from_y)

        found = np.array([False], dtype=np.bool_)
        max_iterations = self.width * self.height * 2

        for _ in range(max_iterations):
            cuda.memcpy_htod(self.gpu_buffers['found'], np.array([False], dtype=np.bool_))

            if self.memory_manager.needs_partitioning:
                for partition_start in range(0, self.memory_manager.total_size,
                                          self.memory_manager.partition_size):
                    partition_size = min(self.memory_manager.partition_size,
                                      self.memory_manager.total_size - partition_start)
                    self.process_partition(partition_start, partition_size, start, goal)
            else:
                self.process_partition(0, self.memory_manager.total_size, start, goal)

            cuda.memcpy_dtoh(found, self.gpu_buffers['found'])
            cuda.memcpy_dtoh(open_set, self.gpu_buffers['open_set'])

            if found[0] or not np.any(open_set):
                break

        # Obtener el número total de estados expandidos
        expanded_states = np.zeros(1, dtype=np.int32)
        cuda.memcpy_dtoh(expanded_states, self.gpu_buffers['expanded_states'])
        self.expanded_states = int(expanded_states[0])

        # Reconstruir camino
        cuda.memcpy_dtoh(came_from_x, self.gpu_buffers['came_from_x'])
        cuda.memcpy_dtoh(came_from_y, self.gpu_buffers['came_from_y'])

        if not found[0]:
            return None

        path = []
        current = goal
        while current != start:
            path.append(current)
            current_idx = current[1] * self.width + current[0]
            next_x = came_from_x[current_idx]
            next_y = came_from_y[current_idx]
            if next_x == -1 or next_y == -1:
                if current == goal:
                    return None
                break
            current = (next_x, next_y)
        path.append(start)
        return path[::-1]

    def get_expanded_states(self) -> int:
        """
        Retorna el número de estados expandidos durante la última búsqueda.

        Returns:
            int: Número de estados expandidos
        """
        return self.expanded_states

    def __del__(self):
        """Liberar recursos de GPU al destruir la instancia."""
        for buffer in self.gpu_buffers.values():
            buffer.free()

def generate_random_obstacles(width: int, height: int, start: Tuple[int, int],
                            goal: Tuple[int, int], obstacle_density: float = 0.3) -> np.ndarray:
    """
    Genera un grid con obstáculos aleatorios, asegurando un camino posible.

    Args:
        width: Ancho del grid
        height: Alto del grid
        start: Tupla (x, y) del punto inicial
        goal: Tupla (x, y) del punto final
        obstacle_density: Porcentaje de celdas que serán obstáculos (0.0 a 1.0)

    Returns:
        numpy.ndarray: Grid con obstáculos aleatorios
    """
    grid = np.zeros((height, width), dtype=np.int32)
    x, y = start
    end_x, end_y = goal

    # Asegurar que inicio y fin estén libres
    safe_positions = set([start, goal])

    # Crear camino garantizado
    while (x, y) != (end_x, end_y):
        if x < end_x and random.random() < 0.5:
            x += 1
        elif x > end_x and random.random() < 0.5:
            x -= 1
        elif y < end_y:
            y += 1
        elif y > end_y:
            y -= 1
        safe_positions.add((x, y))

    # Añadir obstáculos aleatorios
    num_obstacles = int((width * height - len(safe_positions)) * obstacle_density)
    for _ in range(num_obstacles):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        if (x, y) not in safe_positions:
            grid[y, x] = 1

    return grid

def visualize_grid(grid: np.ndarray, path: Optional[List[Tuple[int, int]]] = None) -> str:
    """
    Visualiza el grid y el camino encontrado usando caracteres ASCII.

    Args:
        grid: Array numpy con el grid
        path: Lista opcional de coordenadas del camino encontrado

    Returns:
        str: Representación visual del grid
    """
    display_grid = grid.copy()
    if path:
        for x, y in path:
            if display_grid[y, x] == 0:
                display_grid[y, x] = 2

    symbols = {0: '·', 1: '█', 2: '◊'}
    return '\n'.join(' '.join(symbols[cell] for cell in row) for row in display_grid)

def main():
    """Función principal para demostrar el uso del algoritmo."""
    
    width, height = 2000, 2000
    start = (0, 0)
    goal = (width-1, height-1)
    max_gpu_memory = 128 * 1024 * 1024

    print("Generando grid...")
    grid = generate_random_obstacles(width, height, start, goal, obstacle_density=0.0)

    print("Inicializando A* CUDA...")
    astar = AStarCUDA(grid, max_gpu_memory=max_gpu_memory)

    print("Buscando camino...")
    start_time = time.time()
    path = astar.find_path(start, goal)
    end_time = time.time()

    elapsed_time = end_time - start_time
    expanded_states = astar.get_expanded_states()
    
    # Calcular tasa de expansión (nodos/ms)
    expansion_rate = expanded_states / (elapsed_time * 1000)  # Convertir segundos a milisegundos

    print(f"Tiempo transcurrido: {elapsed_time:.3f} segundos")
    print(f"Número de estados expandidos: {expanded_states:,}")
    print(f"Tasa de expansión: {expansion_rate:.2f} nodos/ms")

    if path:
        print("\nCamino encontrado!")
        print(f"Longitud del camino: {len(path)}")
        #print("\nGrid con camino:")
        #print(visualize_grid(grid, path))
    else:
        print("\nNo se encontró un camino")
        print("\nGrid:")
        print(visualize_grid(grid))

if __name__ == "__main__":
    main()
#include <windows.h>
#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <atomic>
#include <stack>
#include <queue>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <climits>
#include <condition_variable>

using namespace std;

const int THREAD_COUNT = 4;

// Чтение матрицы из файла: первая строка m n, далее m×n чисел по строкам
void read_matrix_from_file(const std::string& filename, std::vector<std::vector<int>>& matrix) {
    ifstream file(filename);
    int m, n;
    file >> m >> n;
    matrix.assign(m, vector<int>(n));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            file >> matrix[i][j];
    file.close();
}

// 1. WinAPI
struct RowMinData {
    const std::vector<std::vector<int>>* matrix;
    int row;
    int* row_mins;
};

DWORD WINAPI FindRowMinWinAPI(LPVOID param) {
    RowMinData* data = (RowMinData*)param;
    const auto& row_vec = (*data->matrix)[data->row];
    int local_min = row_vec.empty() ? INT_MAX : row_vec[0];
    for (int v : row_vec) if (v < local_min) local_min = v;
    data->row_mins[data->row] = local_min;
    return 0;
}

int findLastRowWithMin_WinAPI(const std::vector<std::vector<int>>& matrix) {
    if (matrix.empty()) return -1;
    int n = (int)matrix.size();
    std::vector<int> row_mins(n, INT_MAX);
    std::vector<HANDLE> threads(n);
    std::vector<RowMinData> datas(n);

    for (int i = 0; i < n; ++i) {
        datas[i] = RowMinData{ &matrix, i, row_mins.data() };
        threads[i] = CreateThread(NULL, 0, FindRowMinWinAPI, &datas[i], 0, NULL);
    }
    WaitForMultipleObjects(n, threads.data(), TRUE, INFINITE);
    for (int i = 0; i < n; ++i) CloseHandle(threads[i]);

    int global_min = *std::min_element(row_mins.begin(), row_mins.end());
    for (int i = n - 1; i >= 0; --i)
        if (std::find(matrix[i].begin(), matrix[i].end(), global_min) != matrix[i].end())
            return i;
    return -1;
}

// 2. std::thread
void FindRowMinThread(const std::vector<int>& row, int& result) {
    result = row.empty() ? INT_MAX : row[0];
    for (int v : row) if (v < result) result = v;
}

int findLastRowWithMin_Thread(const std::vector<std::vector<int>>& matrix) {
    if (matrix.empty()) return -1;
    int n = (int)matrix.size();
    std::vector<int> row_mins(n, INT_MAX);
    std::vector<std::thread> threads;
    for (int i = 0; i < n; ++i)
        threads.emplace_back(FindRowMinThread, std::cref(matrix[i]), std::ref(row_mins[i]));
    for (auto& t : threads) t.join();

    int global_min = *std::min_element(row_mins.begin(), row_mins.end());
    for (int i = n - 1; i >= 0; --i)
        if (std::find(matrix[i].begin(), matrix[i].end(), global_min) != matrix[i].end())
            return i;
    return -1;
}

// 3. std::future (async)
int FindRowMinFuture(const std::vector<int>& row) {
    int local_min = row.empty() ? INT_MAX : row[0];
    for (int v : row) if (v < local_min) local_min = v;
    return local_min;
}

int findLastRowWithMin_Future(const std::vector<std::vector<int>>& matrix) {
    if (matrix.empty()) return -1;
    int n = (int)matrix.size();
    std::vector<std::future<int>> futures;
    for (int i = 0; i < n; ++i)
        futures.push_back(std::async(std::launch::async, FindRowMinFuture, std::cref(matrix[i])));
    std::vector<int> row_mins(n);
    for (int i = 0; i < n; ++i) row_mins[i] = futures[i].get();

    int global_min = *std::min_element(row_mins.begin(), row_mins.end());
    for (int i = n - 1; i >= 0; --i)
        if (std::find(matrix[i].begin(), matrix[i].end(), global_min) != matrix[i].end())
            return i;
    return -1;
}

// 4. Interlocked (Windows)
LONG global_min_interlocked_matrix = LONG_MAX;

struct RowMinDataInterlocked {
    const std::vector<int>* row;
};

DWORD WINAPI FindRowMinInterlockedMatrix(LPVOID param) {
    RowMinDataInterlocked* data = (RowMinDataInterlocked*)param;
    int local_min = data->row->empty() ? INT_MAX : (*data->row)[0];
    for (int v : *data->row) if (v < local_min) local_min = v;
    LONG current_min = global_min_interlocked_matrix;
    while (local_min < current_min) {
        LONG old_min = InterlockedCompareExchange(&global_min_interlocked_matrix, local_min, current_min);
        if (old_min == current_min) break;
        current_min = old_min;
    }
    return 0;
}

int findLastRowWithMin_Interlocked(const std::vector<std::vector<int>>& matrix) {
    if (matrix.empty()) return -1;
    int n = (int)matrix.size();
    global_min_interlocked_matrix = LONG_MAX;
    std::vector<HANDLE> threads(n);
    std::vector<RowMinDataInterlocked> datas(n);

    for (int i = 0; i < n; ++i) {
        datas[i] = RowMinDataInterlocked{ &matrix[i] };
        threads[i] = CreateThread(NULL, 0, FindRowMinInterlockedMatrix, &datas[i], 0, NULL);
    }
    WaitForMultipleObjects(n, threads.data(), TRUE, INFINITE);
    for (int i = 0; i < n; ++i) CloseHandle(threads[i]);

    int global_min = global_min_interlocked_matrix;
    for (int i = n - 1; i >= 0; --i)
        if (std::find(matrix[i].begin(), matrix[i].end(), global_min) != matrix[i].end())
            return i;
    return -1;
}

// 5. Thread-safe stack with WinAPI Event synchronization

struct Task {
    int start;
    int end;
};

class EventThreadSafeStack {
private:
    std::stack<Task> tasks;
    HANDLE hEvent; // Событие для сигнализации о наличии задач
    CRITICAL_SECTION cs; // Критическая секция для защиты стека
public:
    EventThreadSafeStack() {
        hEvent = CreateEvent(NULL, TRUE, FALSE, NULL); // manual-reset, nonsignaled
        InitializeCriticalSection(&cs);
    }
    ~EventThreadSafeStack() {
        CloseHandle(hEvent);
        DeleteCriticalSection(&cs);
    }
    void push(const Task& task) {
        EnterCriticalSection(&cs);
        tasks.push(task);
        SetEvent(hEvent); // Сигнализируем о наличии задачи
        LeaveCriticalSection(&cs);
    }
    // Возвращает true если задача получена, false если стек пуст
    bool pop(Task& task) {
        EnterCriticalSection(&cs);
        if (tasks.empty()) {
            ResetEvent(hEvent); // Нет задач — сбрасываем событие
            LeaveCriticalSection(&cs);
            return false;
        }
        task = tasks.top();
        tasks.pop();
        if (tasks.empty()) ResetEvent(hEvent); // Если после pop стек пуст — сбрасываем событие
        LeaveCriticalSection(&cs);
        return true;
    }
    HANDLE getEventHandle() const { return hEvent; }
    bool empty() {
        EnterCriticalSection(&cs);
        bool res = tasks.empty();
        LeaveCriticalSection(&cs);
        return res;
    }
};

// Глобальные переменные для задачи 5
EventThreadSafeStack* eventTaskStackPtr;
const std::vector<std::vector<int>>* arrPtrEventStack;
int global_min_event_stack = INT_MAX;

// Воркер для пула потоков 
DWORD WINAPI ThreadPoolWorkerEventStack(LPVOID param) {
    Task task;
    while (true) {
        WaitForSingleObject(eventTaskStackPtr->getEventHandle(), INFINITE);
        if (!eventTaskStackPtr->pop(task)) {
            continue; // Ждём следующую задачу
        }
        if (task.start == -1) break; // Специальная задача для завершения
        int row = task.start;
        int local_min = (*arrPtrEventStack)[row].empty() ? INT_MAX : (*arrPtrEventStack)[row][0];
        for (int v : (*arrPtrEventStack)[row]) if (v < local_min) local_min = v;
        LONG current_min = global_min_event_stack;
        while (local_min < current_min) {
            LONG old_min = InterlockedCompareExchange((volatile LONG*)&global_min_event_stack, local_min, current_min);
            if (old_min == current_min) break;
            current_min = old_min;
        }
    }
    return 0;
}

int findLastRowWithMin_EventStack(const std::vector<std::vector<int>>& matrix) {
    if (matrix.empty()) return -1;
    int n = (int)matrix.size();
    global_min_event_stack = INT_MAX;
    EventThreadSafeStack taskStack;
    eventTaskStackPtr = &taskStack;
    arrPtrEventStack = &matrix;

    // Кладём задачи (по одной на каждую строку)
    for (int i = 0; i < n; ++i) {
        Task task;
        task.start = i;
        task.end = i + 1;
        taskStack.push(task);
    }

    // Запускаем потоки
    std::vector<HANDLE> threads(THREAD_COUNT);
    for (int i = 0; i < THREAD_COUNT; ++i) {
        threads[i] = CreateThread(NULL, 0, ThreadPoolWorkerEventStack, NULL, 0, NULL);
    }

    // Ждём, пока все обычные задачи будут обработаны
    while (!taskStack.empty()) {
        Sleep(1);
    }

    // Кладём задачи-завершения
    for (int i = 0; i < THREAD_COUNT; ++i) {
        Task endTask;
        endTask.start = -1;
        endTask.end = -1;
        taskStack.push(endTask);
    }
    SetEvent(taskStack.getEventHandle());

    WaitForMultipleObjects(THREAD_COUNT, threads.data(), TRUE, INFINITE);
    for (int i = 0; i < THREAD_COUNT; ++i) CloseHandle(threads[i]);

    int global_min = global_min_event_stack;
    for (int i = n - 1; i >= 0; --i)
        if (std::find(matrix[i].begin(), matrix[i].end(), global_min) != matrix[i].end())
            return i;
    return -1;
}



// 6. Producer-Consumer
int findLastRowWithMin_ProducerConsumer(const std::vector<std::vector<int>>& matrix) {
    if (matrix.empty()) return -1;
    int n = (int)matrix.size();
    std::queue<int> taskQueue;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<int> row_mins(n, INT_MAX);
    bool done = false;

    auto producer = [&]() {
        for (int i = 0; i < n; ++i) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                taskQueue.push(i);
            }
            cv.notify_one();
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            done = true;
        }
        cv.notify_all();
        };

    auto consumer = [&]() {
        while (true) {
            int idx = -1;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&]() { return !taskQueue.empty() || done; });
                if (taskQueue.empty() && done) break;
                if (!taskQueue.empty()) {
                    idx = taskQueue.front();
                    taskQueue.pop();
                }
            }
            if (idx != -1) {
                int local_min = matrix[idx].empty() ? INT_MAX : matrix[idx][0];
                for (int v : matrix[idx]) if (v < local_min) local_min = v;
                row_mins[idx] = local_min;
            }
        }
        };

    std::thread prod(producer);
    std::thread cons1(consumer), cons2(consumer);
    prod.join(); cons1.join(); cons2.join();

    int global_min = *std::min_element(row_mins.begin(), row_mins.end());
    for (int i = n - 1; i >= 0; --i)
        if (std::find(matrix[i].begin(), matrix[i].end(), global_min) != matrix[i].end())
            return i;
    return -1;
}

// 7. OpenMP
int findLastRowWithMin_OpenMP(const std::vector<std::vector<int>>& matrix) {
    if (matrix.empty()) return -1;
    int n = (int)matrix.size();
    std::vector<int> row_mins(n, INT_MAX);

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        int local_min = matrix[i].empty() ? INT_MAX : matrix[i][0];
        for (int v : matrix[i]) if (v < local_min) local_min = v;
        row_mins[i] = local_min;
    }

    int global_min = *std::min_element(row_mins.begin(), row_mins.end());
    int last_row = -1;
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (std::find(matrix[i].begin(), matrix[i].end(), global_min) != matrix[i].end()) {
#pragma omp critical
            if (i > last_row) last_row = i;
        }
    }
    return last_row;
}

int main() {
    SetConsoleOutputCP(1251);
    std::vector<std::vector<int>> matrix;
    read_matrix_from_file("data.txt", matrix);
    std::cout << "Прочитано строк: " << matrix.size() << std::endl;

    std::cout << "WinAPI: " << findLastRowWithMin_WinAPI(matrix) << std::endl;
    std::cout << "std::thread: " << findLastRowWithMin_Thread(matrix) << std::endl;
    std::cout << "std::future: " << findLastRowWithMin_Future(matrix) << std::endl;
    std::cout << "Interlocked: " << findLastRowWithMin_Interlocked(matrix) << std::endl;
    std::cout << "Event ThreadSafeStack: " << findLastRowWithMin_EventStack(matrix) << std::endl;
    std::cout << "Producer-Consumer: " << findLastRowWithMin_ProducerConsumer(matrix) << std::endl;
    std::cout << "OpenMP: " << findLastRowWithMin_OpenMP(matrix) << std::endl;

    return 0;
}
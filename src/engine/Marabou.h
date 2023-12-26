/*********************                                                        */
/*! \file Marabou.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

 **/

#ifndef __Marabou_h__
#define __Marabou_h__

#include "AcasParser.h"
#include "OnnxParser.h"
#include "Engine.h"
#include "InputQuery.h"
#include "minisat/core/Solver.h"
#include "thread"
class Marabou
{
public:
    Marabou();
    ~Marabou();

    /*
      Entry point of this class
    */
    void run();

protected:
    InputQuery _inputQuery;
    /*
      Extract the options and input files (network and property), and
      use them to generate the input query
    */
    void prepareInputQuery();
    void extractSplittingThreshold();

    /*
      Invoke the engine to solve the input query
    */
    void solveQuery();

    /*
      Display the results
    */
    void displayResults( unsigned long long microSecondsElapsed ) const;

    /*
      Export assignment as per Options
     */
    void exportAssignment() const;

    /*
      Import assignment for debugging as per Options
     */
    void importDebuggingSolution();

    /*
      ACAS network parser
    */
    AcasParser *_acasParser;

    /*
      ONNX network parser
    */
    OnnxParser *_onnxParser;

    /*
      The solver
    */
protected:
    Engine _engine;

    Minisat::Solver _solver;

};

class ThreadSafeQueue {
private:
    std::vector<int> data_queue;
    mutable std::mutex mut;
    std::condition_variable data_cond;
public:
    void push(int new_value) {
        std::lock_guard<std::mutex> lk(mut);
        data_queue.push_back(new_value);
        data_cond.notify_one();
    }

    void wait_and_pop(int& value) {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk,[this]{return !data_queue.empty();});
        value=data_queue.back();
        data_queue.pop_back();
    }
};

class test: public Marabou {
public:
    test() : Marabou() {
        printf("test created\n");
    }
};

class FinderWorker : public Marabou {
private:
    ThreadSafeQueue& _taskQueue;
    std::atomic<int>& _learnt_count, &_finish_worker_count;
    SearchPath& _searchPath;
    int _id;
    int _last_task_index = -1;
public:
    FinderWorker(ThreadSafeQueue& taskQueue, std::atomic<int>& learnt_count, std::atomic<int>& finish_worker_count, SearchPath& searchPath, int worker_id=0) :
    Marabou(), _taskQueue(taskQueue), _searchPath(searchPath), _learnt_count(learnt_count),
    _finish_worker_count(finish_worker_count),_id(worker_id) {
        printf("FinderWorker %d created\n", _id);
        init();
    }

    void init() {
        prepareInputQuery();
        if (_engine.processInputQuery( _inputQuery )) {
            _engine.simpleInitEngine();
        }
    }

    ~FinderWorker() {
        printf("FinderWorker %d destroyed\n", _id);
    }

    void LearnClause(int index) {
        std::vector<PathElement> path;
        auto old_path = std::move(_searchPath.getPath(index));
        if (index % 20 == 0) {
            dumpSearchPath(old_path);
        }
        auto learn = _engine.conflictClauseLearningWithRestore(old_path, path);
        if (learn) {
            printf("Worker %d learnt clauses from the [%d]-th path, size from [%zu] to [%zu]\n", _id, index,
                   old_path.size(), path.size());
            if (path.size() < old_path.size()) {
                _searchPath.safeAppendLearned(path);
                _learnt_count++;
            }
        }
        printf("Worker %d finished learning from [%d]-th path\n", _id, index);
    }

    void start() {
        while (true) {
            int index;
            _taskQueue.wait_and_pop(index);
            if (index == -1) {
                printf("Worker %d exit\n", _id);
                _finish_worker_count++;
                break;
            }
            if (index < _last_task_index) {
                printf("Worker %d skip task %d\n", _id, index);
                continue;
            }
            _last_task_index = index;
            LearnClause(index);
        }
    }
};

class ConflictClauseFinder {
private:

    ThreadSafeQueue _taskQueue;
    SearchPath& _searchPath;
    int _num_workers;
    std::vector<std::shared_ptr<FinderWorker>> _workers;
    std::vector<std::thread> _worker_threads;
    std::atomic<int> _learnt_count, _finish_worker_count;
public:
    ConflictClauseFinder(SearchPath& searchPath, int num_workers=6) :
            _searchPath(searchPath), _num_workers(num_workers), _learnt_count(0), _finish_worker_count(0){
        for (int i = 0; i < num_workers; ++ i) {
            auto worker = std::make_shared<FinderWorker>(_taskQueue, _learnt_count, _finish_worker_count, _searchPath, i);
            _workers.emplace_back(worker);
            _worker_threads.emplace_back(std::thread(&FinderWorker::start, worker));
        }
        start();
    }

    void start() {
        for (auto& worker : _worker_threads) {
            worker.detach();
        }
    }

    void addTask(int index) {
        printf("add task %d\n", index);
        _taskQueue.push(index);
    }

    int getLearntCount() {
        return _learnt_count;
    }

    void stop() {
        for (int i = 0; i < _num_workers; i++) {
            _taskQueue.push(-1);
        }
        while (_finish_worker_count < _num_workers) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

};


#endif // __Marabou_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//

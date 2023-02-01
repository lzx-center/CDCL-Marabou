import multiprocessing
import random, time

def worker(id, msg):
  start = time.time()
  time.sleep(5)
  end = time.time()
  print(f'{id} : {msg} use time {end-start}')


if __name__ == "__main__": 
  pool = multiprocessing.Pool(processes = 8)
  start = time.time()
  for i in range(5):
    pool.apply_async(worker, (i, f"test  {i}"))
  
  pool.close()
  pool.join()
  # for i in range(5):
  #   worker(i, f"test  {i}")
  # end = time.time()
  # print(end - start)
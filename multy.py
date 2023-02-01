from multiprocessing import Pool

def run_lastz(file_tuple):
    f1,f2,out = file_tuple
    print " Working with input -",f1,f2,out
    print "                 "

if __name__ == '__main__':
    # Start 10 worker processes
    pool = Pool()

    #Then create a list of input list [(1.txt,2.txt,1_2.out),(2.txt,3.txt,2_3.out)....]
    file_name_list = ['1.txt', '2.txt', '3.text', '15.txt', '111.txt', '31.txt', '41.txt', '50.txt', '1011.txt']
    f_list =[]
    for file1 in file_name_list:
        for file2 in file_name_list:
            if file1 != file2:
                out_file = '{}__{}.out'.format(file1, file2)
                t_list =(file1,file2,out_file)
                f_list.append(t_list)
    #Map input list with target function
    pool.map(run_lastz, f_list)
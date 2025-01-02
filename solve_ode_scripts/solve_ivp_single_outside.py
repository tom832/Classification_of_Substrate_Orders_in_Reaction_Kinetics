from scipy.integrate import solve_ivp
import random
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
import os
import argparse
import json


def mechanism(t, z, A, a, B, C, c):
    s, p, bp, cat, cats= z
    return (a*cats-A*cat*s-C*s+c*bp, B*cats, C*s-c*bp, -A*cat*s+a*cats+B*cats, A*cat*s-a*cats-B*cats)

def solve_task(A, a, B, C, c):
    sol = solve_ivp(mechanism, [0, 180], [1, 0, 0, 0.01, 0], args=(A, a, B, C, c), dense_output=True)
    t = np.linspace(0, 180, 30)
    z = sol.sol(t)
    if z[0][19] > 0.05 and z[1][29] > 0.5 and 0.05 < z[2][29] < 0.5:
        return t, z, A, a, B, C, c
    else:
        return None

def log_uniform(lower_bound, upper_bound):
    log_lower_bound = np.log(lower_bound)
    log_upper_bound = np.log(upper_bound)
    
    log_random = random.uniform(log_lower_bound, log_upper_bound)
    
    return np.exp(log_random)


if __name__ == '__main__':

    # python solve_ivp_single_outside.py --task_name 0 --task_id hpc1 --job_num 100000 --num_proc 5
    parser = argparse.ArgumentParser(description='Solve IVP Script')
    parser.add_argument('--task_name', type=str, default='0', help='task name')
    # parser.add_argument('--task_id', type=str, default='0', help='task id')
    parser.add_argument('--job_num', type=int, default=50000, help='job num')
    parser.add_argument('--num_proc', type=int, default=4, help='num proc')
    args = parser.parse_args()
    task_name = args.task_name
    # task_id = args.task_id
    job_num = args.job_num
    num_proc = args.num_proc


    tasks = []
    # save_root = f'{task_name}__single_outside__{task_id}'
    save_root = os.path.join('../data', 'ode_raw_data', f'{task_name}__single_outside')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    start_time = time.time()
    data_cnt = len(os.listdir(save_root))
    print('start from', data_cnt)


    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        for _ in range(job_num): 
            
            a = random.uniform(1, 10)*10**random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
            B = random.uniform(1, 10)*10**random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])

            if task_name == '1':
                A = log_uniform(10**(-5), (a+B)/10)
                while 0.01*A*B/(a+B)<10**(-5):
                    a = random.uniform(1, 10)*10**random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
                    B = random.uniform(1, 10)*10**random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
                    A = log_uniform(10**(-5), (a+B)/10)
                
            elif task_name == 'mm':
                A_down_bound = (a+B)/10
                A_up_bound = (a+B)/0.02
                if A_down_bound < 10**(-5):
                    A_down_bound = 10**(-5)
                if A_up_bound > 10**5:
                    A_up_bound = 10**5
                A = log_uniform(A_down_bound, A_up_bound)
                while 0.01*A*B/(a+B)<10**(-5):
                    a = random.uniform(1, 10)*10**random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
                    B = random.uniform(1, 10)*10**random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
                    A_down_bound = (a+B)/10
                    A_up_bound = (a+B)/0.02
                    if A_down_bound < 10**(-5):
                        A_down_bound = 10**(-5)
                    if A_up_bound > 10**5:
                        A_up_bound = 10**5
                    A = log_uniform(A_down_bound, A_up_bound)

            elif task_name == '0':
                B = log_uniform(0.28, 0.79) # it's a reasonable range for B to make the task solvable
                while (a+B)/0.005 > 10**5:
                    a = random.uniform(1, 10)*10**random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
                    B = log_uniform(0.28, 0.79)
                A = log_uniform((a+B)/0.005, 10**5)
            
            C_up_bound = 0.01*A*B/(a+B)
            C_down_bound = 10**(-5)
            if C_up_bound > 10**(5):
                C_up_bound = 10**(5)            
            C = log_uniform(10**(-5), C_up_bound)
            c = log_uniform(10**(-5), C)

            tasks.append(executor.submit(solve_task, A, a, B, C, c))

        for future in tasks:
            result = future.result()
            if result:
                t, z, A, a, B, C, c = result
                save_path = os.path.join(save_root, f'{str(data_cnt)}.json')
                with open(save_path,'w') as f:
                    json.dump({"t":np.round(t, 3).tolist(), "s":np.round(z[0], 6).tolist(), "p":np.round(z[1], 6).tolist(), "bp":np.round(z[2], 6).tolist(), "cat":np.round(z[3], 6).tolist(), "cats":np.round(z[4], 6).tolist(), "A":A, "a":a, "B":B, "C":C, "c":c}, f, indent=2)

                data_cnt += 1
                if data_cnt % 100 == 0:
                    print(data_cnt, time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))


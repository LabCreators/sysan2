from task_solution import *

a = Solve({
	'samples': 50,
	'input_file': 'input_own.txt',
	'dimensions': [2, 4, 4, 1],
	'output_file': 'output.txt', 
	'degrees': [3, 3, 3],
     	'lambda_multiblock': False, 
	'weights': 'scaled', 
	'poly_type': 'hermit'
	})

a.define_data()
a.norm_data()
a.define_norm_vectors()
a.built_B()
a.poly_func()

def test_p(a,p1,p2,p3):
    d = list()
    for i in range(1,p1):
        for j in range(1,p2):
            for k in range(1,p3):
                a.p = [i+1,j+1,k+1]
                print(a.p)
                a.built_A()
                a.lamb()
                a.psi()
                a.built_a()
                a.built_Fi()
                a.built_c()
                a.built_F()
                a.built_F_()
                d.append((str(i)+' '+str(j)+' '+str(k),np.linalg.norm(a.norm_error)))
    return d

d = test_p(a,13,13,13)
f = open('test_p_own.txt','w')
miner = d[0]
for i in d:
    f.write(str(i[0])+' : '+str(i[1]))
    f.write('\n')
    if i[1] < miner[1]:
        miner = i
print(miner)
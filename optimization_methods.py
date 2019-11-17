import numpy as np
import itertools


def gradient(func, h, params):
    """
    calculates a gradient of a given function by numerical methods
    """
    gradient_vector = np.array([(func(*(params[:i]+[params[i]+h]+params[i+1:]))-
                          func(*(params[:i]+[params[i]-h]+params[i+1:])))/(2*h) for i in range(len(params))])
    return gradient_vector


def gessian(func, h, params):
    """
    calculates a gessian of a given function by numerical methods
    """
    hesse_matrix = np.zeros((len(params), len(params)))
    for i in range(len(params)):
        for j in range(i, len(params)):
            if i==j:
                hesse_matrix[i][j] = (func(*(params[:i] + [params[i]+h] + params[i+1:]))-\
                                      2*func(*params)+func(*(params[:i] + [params[i]-h] + params[i+1:])))/(pow(h,2))
            else:
                hesse_matrix[i][j]=((func(*(params[:i]+[params[i]+h]+params[i+1:j]+[params[j]+h]+params[j+1:]))-\
                                   func(*(params[:i]+[params[i]-h]+params[i+1:j]+[params[j]+h]+params[j+1:])))-\
                                   (func(*(params[:i]+[params[i]+h]+params[i+1:j]+[params[j]-h]+params[j+1:]))-\
                                   func(*(params[:i]+[params[i]-h]+params[i+1:j]+[params[j]-h]+params[j+1:])))
                                   )/(4*pow(h,2))
        for j in range(i):
            hesse_matrix[i][j]=hesse_matrix[j][i]

    return hesse_matrix

def hyperplane_projection(params, args):
    """
    calculates projection for hyperplane
    """
    beta, coefs = args
    return np.array(params) + np.dot((beta - np.dot(np.array(coefs), np.array(params))), 
                                     np.array(coefs)/pow(np.linalg.norm(np.array(coefs)), 2))

def sphere_projection(params, args):
    """
    calculates projection for sphere
    """
    radius, center = args
    return np.array(center)+ radius*((np.array(params)-np.array(center)
                                    )/np.linalg.norm(np.array(params)-np.array(center)))

def subspace_pojection(params, args):
    """
    calculates projection for subspace
    """
    beta, coefs = args
    return np.array(params) + np.dot(max(0, beta-np.dot(np.array(coefs), np.array(params))), 
                                         np.array(coefs)/pow(np.linalg.norm(np.array(coefs)), 2))

def poliedr_projection(params, args):
    """
    calculates projection for poliedr
    """
    left, right = args
    result = np.array([el for el in range(len(params))])
    result[np.where(np.array(params)<np.array(left))[0]] = np.array(left)[np.where(np.array(params)<np.array(left))[0]]
    result[np.where((np.array(left)<np.array(params)) & (np.array(params)<np.array(right)))[0]] = np.array(params)[
        np.where((np.array(left)<np.array(params)) & (np.array(params)<np.array(right)))[0]]
    result[np.where(np.array(right)<np.array(params))[0]] = np.array(right)[np.where(np.array(right)<np.array(params))[0]]
    return result

def non_negative_orthant_projection(params, args=None):
    """
    calculates projection for non_negative orthant
    """
    return np.array([max(0, el) for el in params])

def fib(n):
    """
    calculates the element of fibonacci sequence by given number
    """
    if (n==1 or n==2):
        return 1
    else:
        return fib(n-2)+fib(n-1)
    
def get_number_fibonacci(start, end, eps):
    for n in range(1, 100000):
        if (fib(n)>=(end-start)/eps):
            break
    return n


def fibonacci_method(func, start, end, eps):
    """
    fibonacci method to minimize a function of one argument
    """
    n = get_number_fibonacci(start, end, eps)
    fn = fib(n)
    x1 = start + (fib(n-2) * (end-start))/fib(n)
    x2 = start + (fib(n-1) * (end-start))/fib(n)
    
    while (end-start>eps):
        if func(x1)<=func(x2):
            end = x2
            x2 = x1
            x1 = start + (fib(n-3) * (end-start))/fib(n)
        else:
            start = x1
            x1 = x2
            x2 = start + (fib(n-2) * (end-start))/fib(n)
    return (start+end)/2


def golden_ratio_method(func, start, end, eps):
    """
    golden ration method to minimize a function of one argument
    """
    fi = (1+pow(5.0, 0.5))/2
    
    x1 = end-(end-start)/fi
    x2 = start + (end-start)/fi
    
    while((end-start)/2>=eps):
        if func(x1)>=func(x2):
            start = x1
            x1 = x2
            x2 = end - x1 + start
        else:
            end = x2
            x2 = x1
            x1 = start + end - x2
    return (start + end)/2


def gradient_descent_swift(func, params, eps, method, start=0, end=1):
    """
    swift gradient descent method to minimize a given function with given method (fibonacci or golden ratio)
    """
    qty_steps=1
    
    dot0 = np.array(params)
    steps = [dot0]
    
    while(np.linalg.norm(gradient(func, eps, dot0.tolist()))>eps):
        
        f_alpha = lambda alpha: func(*(dot0-alpha*gradient(func, eps, dot0.tolist())))
        if method == 'fibonacci':
            step = fibonacci_method(f_alpha, start, end, eps)
        else:
            step = golden_ratio_method(f_alpha, start, end, eps)
            
        dot1 = dot0 - step*gradient(func, eps, dot0.tolist())
        dot0 = dot1
        steps.append(dot0)
        qty_steps+=1

        print('number of iteration: {}, current point: {}, function value: {}'.format(qty_steps, 
                                                                                      dot1, func(*dot1)))
    
    print("Precision: {}".format(np.linalg.norm(gradient(func, eps, dot1.tolist()))))
    
    return steps


def gradient_descent_constant_step(func, params, eps, step):
    """
    gradient descent method to minimize a given function with a constant step
    """
    qty_steps=1
    
    dot0 = np.array(params)
    steps = [dot0]
    dot1 = dot0-step*gradient(func, eps, dot0.tolist())
    
    while(np.linalg.norm(dot1-dot0)>eps):
        
        dot0 = dot1

        steps.append(dot0)

        dot1 = dot0 - step*gradient(func, eps, dot0.tolist())
        qty_steps+=1

        print('number of iteration: {}, current point: {}, function value: {}'.format(qty_steps, 
                                                                                      dot1, func(*dot1)))
    
    print("Precision: {}".format(np.linalg.norm(dot1-dot0)))
    
    return steps


def gradient_descent(func, params, eps):
    """
    gradient descent method to minimize a given function
    """
    qty_steps=1
    step=1
    
    dot0 = np.array(params)
    steps = [dot0]
    
    while(np.linalg.norm(gradient(func, eps, dot0.tolist()))>eps):
        f0 = func(*dot0)
        while(func(*(dot0-step*gradient(func, eps, dot0.tolist())))<f0):
            step*=2
        while(func(*(dot0-step*gradient(func, eps, dot0.tolist())))>=f0):
            step*=0.5
        
        dot1 = dot0-step*gradient(func, eps, dot0.tolist())
        dot0 = dot1
        steps.append(dot0)
        qty_steps+=1
        
        print('number of iteration: {}, current point: {}, function value: {}'.format(qty_steps,
                                                                                      dot1, func(*dot1)))
    
    print("Precision: {}".format(np.linalg.norm(gradient(func, eps, dot1.tolist()))))
    
    return steps


def newton_method(func, params, eps):
    """
    Newton method to minimize a given function
    """
    qty_steps=1
    
    dot0 = np.array(params)
    steps = [dot0]
    dot1 = dot0-np.dot(gradient(func, eps, dot0.tolist()), np.linalg.inv(gessian(func, eps, dot0.tolist())))
    
    while(np.linalg.norm(dot1-dot0)>eps):
        dot0 = dot1

        steps.append(dot0)

        dot1 = dot0 - np.dot(gradient(func, eps, dot0.tolist()), np.linalg.inv(gessian(func, eps, dot0.tolist())))
        qty_steps+=1

        print('number of iteration: {}, current point: {}, function value: {}'.format(qty_steps, 
                                                                                   dot1, func(*dot1)))
    print("Precision: {}".format(np.linalg.norm(dot1-dot0)))
    
    return steps


def gradient_projection(func, eps, projection_function, params, projection_func_args):
    """
    gradient projection method to minimize a target function with given projection function
    """
    qty_steps=1
    step = 1
    
    dot0 = np.array(params)
    steps = [dot0]
    
    f_alpha = lambda alpha: func(*(dot0-alpha*gradient(func, eps, dot0.tolist())))
        
    step = golden_ratio_method(f_alpha, 0, step, eps)
    
    dot1 = projection_function(dot0-step*gradient(func, eps, dot0.tolist()), 
                                 projection_func_args)
    
    while(np.linalg.norm(dot1-dot0)>=eps):
    
        dot0 = dot1
        
        f_alpha = lambda alpha: func(*(dot0-alpha*gradient(func, eps, dot0.tolist())))
        
        step = golden_ratio_method(f_alpha, 0, step, eps)
        
        dot1 = projection_function(dot0-step*gradient(func, eps, dot0.tolist()), 
                                 projection_func_args)
        steps.append(dot0)
        qty_steps+=1
        
        print('number of iteration: {}, current point: {}, function value: {}'.format(qty_steps,
                                                                                      dot1, func(*dot1)))
    
    print("Precision: {}".format(np.linalg.norm(dot1-dot0)))
    
    return steps


def conjucate_gradients_method(func, params, eps, start=0, end=1, quadratic=True):
    """
    general conjucate gradients method to minimize a given function
    """
    iteration_data = []
    qty_steps=1
    step=1
    
    dot0 = np.array(params)
    steps = [dot0]
    h = -gradient(func, eps, dot0.tolist())
    prev = h
    prev_grad = gradient(func, eps, dot0.tolist())
    
    f_alpha = lambda alpha: func(*(dot0+alpha*h))
    step = golden_ratio_method(f_alpha, start, end, eps)
    dot1 = dot0 + step*h
    
    while(np.linalg.norm(dot1 - dot0)>eps):
        
        dot0 = dot1
        
        h = -gradient(func, eps, dot0.tolist()) + np.dot(prev,
                        pow(np.linalg.norm(gradient(func, eps, dot0.tolist())
                      ), 2)/pow(np.linalg.norm(prev_grad), 2))
        prev = h
        prev_grad = gradient(func, eps, dot0.tolist())
        
        #dot0 = dot1
        
        f_alpha = lambda alpha: func(*(dot0+alpha*h))
        step = golden_ratio_method(f_alpha, start, end, eps)
        
        dot1 = dot0+step*h
        
        steps.append(dot0)
        
        qty_steps+=1
        
        if not quadratic:
            print('number of iteration: {}, current point: {}, function value: {}'.format(qty_steps, 
                                                                                      dot1, func(*dot1)))
        else:
            iteration_data.append((qty_steps, dot1, func(*dot1)))
    
    for i, el in enumerate(iteration_data[-3:]):
        print('number of iteration: {}, current point: {}, function value: {}'.format(i+1, el[1], el[2]))
        
    print("Precision: {}".format(np.linalg.norm(dot1-dot0)))
    
    return steps


def conjugate_gradient_method(A, b, eps, maxiterations=100):
    '''
    Conjugate Gradient Method that solve equation Ax = b with given accuracy
    :param A:matrix A
    :param b:vector b
    :param eps: accuracy
    :return: solution x
    '''
    n = len(A.T) # number column
    xi1 = xi = np.zeros(shape=(n,1), dtype = float)
    vi = ri = b # start condition
    i = 0 #loop for number iteration
    while True:
        try:
            i+= 1
            ai = float(vi.T*ri)/float(vi.T*A*vi) # alpha i
            xi1 = xi+ai*vi # x i+1
            ri1 = ri-ai*A*vi # r i+1
            betai = -float(vi.T*A*ri1)/float(vi.T*A*vi) # beta i
            vi1 = ri1+betai*vi
            if (np.linalg.norm(ri1)<eps) or i > 10 * n or i > maxiterations:
                break
            else:
                xi,vi,ri = xi1,vi1,ri1
        except Exception:
            print("There is a problem with minimization.")
    return np.matrix(xi1)

def conjucate_grads(A, b, x=None):
    n = len(b)
    if isinstance(x, type(None)):
        x = np.ones(n)
    r = np.dot(A, x) - b
    p = - r
    r_k_norm = np.dot(r, r)
    for i in range(2*n):
        Ap = np.dot(A, p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-5:
            print('Itr:', i)
            break
        p = beta * p - r
    return x


def coordinate_descent(A, b, eps, maxIterations = 100):
    A = np.array(A)
    N = A.shape[0]
    x = b #[0 for i in range(N)]
    xprev = [0.0 for i in range(N)]
    norm = N
    it = 0
    while norm > eps or it < maxIterations:
        for j in range(N):
            xprev[j] = x[j]
        for j in range(N):
            summ = 0.0
            for k in range(N):
                if (k != j):
                    summ = summ + A[j][k] * x[k]
            x[j] = (b[j] - summ) / A[j][j]
        diff1norm = 0.0
        oldnorm = 0.0
        for j in range(N):
            diff1norm = diff1norm + abs(x[j] - xprev[j])
            oldnorm = oldnorm + abs(xprev[j])
        if oldnorm == 0.0:
            oldnorm = 1.0
        norm = diff1norm / oldnorm
        it+=1
    return x


def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix,weights)
    return(predictions)

def lasso_coordinate_descent_step(num_features, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix, weights)
    #z_i= (feature_matrix*feature_matrix).sum()

    for i in range(num_features+1):
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
        ro_i = (feature_matrix[:,i]*(output - prediction+weights[i]*feature_matrix[:,i])).sum()
        if i == 0: # intercept -- do not regularize
            new_weight_i = ro_i
        elif ro_i < -l1_penalty/2.:
            new_weight_i = (ro_i +(l1_penalty/2))
        elif ro_i > l1_penalty/2.:
            new_weight_i = (ro_i -(l1_penalty/2))
        else:
            new_weight_i = 0.
    return new_weight_i


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights,
                                      l1_penalty, tolerance):
    condition = True
    max_change=0
    while (condition == True):
        max_change = 0
        for i in range(len(initial_weights)):
            #max_change=0
            old_weight_i= initial_weights[i]
            initial_weights[i] =lasso_coordinate_descent_step(i,
                                                      feature_matrix, output,
                                                      initial_weights, l1_penalty)
            coordinate_change = abs(old_weight_i - initial_weights[i])
            if coordinate_change > max_change:
                max_change = coordinate_change
        if (coordinate_change < tolerance):
            condition = False
    return initial_weights

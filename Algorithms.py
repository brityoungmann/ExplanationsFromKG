from pyitlib import discrete_random_variable as drv
import sys
import numpy as np

EPSILON = 0.01


def MCIMR(E, x, y, num, verbose = False):
    results = []
    X = [int(ee) for ee in x]
    Y = [int(ee) for ee in y]

    print(drv.information_mutual(X,Y))

    # first iteration
    for e in E:
        E[e] = E[e].fillna(-1)
        varE = E[e].tolist()
        varE = [int(ee) for ee in varE]
        count = varE.count(-1)
        if count>0:
            indices = [i for i, x in enumerate(varE) if x == -1]
            varEE = [x for i, x in enumerate(varE) if not i in indices]
            XX = [x for i, x in enumerate(X) if not i in indices]
            YY = [x for i, x in enumerate(Y) if not i in indices]
            if len(XX) == 0:
                continue
            valE = drv.information_mutual_conditional(XX, YY, varEE)
        else:
            valE = drv.information_mutual_conditional(X, Y, varE)
        results.append((e,valE))
        print(e,valE)

    results = sorted(results, key=lambda x: x[1])
    if verbose:
        print(results)
    ans =set()
    ans.add(results[0][0])
    EE = [e for e in E if not e in ans]

    for i in range(num-1):
        opt = sys.maxsize
        toAdd = None
        for e in EE:

            varE = E[e].tolist()
            varE = [int(ee) for ee in varE]
            count = varE.count(-1)
            if count > 0:
                indices = [i for i, x in enumerate(varE) if x == -1]
                varEE = [x for i, x in enumerate(varE) if not i in indices]
                XX = [x for i, x in enumerate(X) if not i in indices]
                YY = [x for i, x in enumerate(Y) if not i in indices]
                if len(XX) == 0:
                  continue
                t1 = drv.information_mutual_conditional(XX, YY, varEE)
            else:
                t1 = drv.information_mutual_conditional(X, Y, varE)
            if t1 > opt:
                continue
            t2 = 0
            tempAns = [x for x in ans]
            tempAns.append(e)
            for e1, e2 in zip(tempAns, tempAns[1:]):
                varE1 = E[e1].tolist()
                indices1 = [i for i, x in enumerate(varE) if x == -1]
                varE2 = E[e2].tolist()
                indices2 = [i for i, x in enumerate(varE2) if x == -1]
                indices = set(indices1+indices2)
                varE1 = [x for i, x in enumerate(varE1) if not i in indices]
                varE2 = [x for i, x in enumerate(varE2) if not i in indices]
                varE1 = [int(ee) for ee in varE1]
                varE2 = [int(ee) for ee in varE2]
                if len(varE1) == 0:
                    continue
                t2 = t2 + drv.information_mutual(varE1, varE2)
            t2 = t2/(len(tempAns)-1)
            val = t1 + t2
            if val < opt:
                opt = t1 + t2
                toAdd = e
        #resposibility test
        ans.add(toAdd)
        Eans = E[list(ans)]
        num = responsibility(X, Y, Eans, toAdd)
        if num <EPSILON:
            ans.remove(toAdd)
            return ans
        EE = [e for e in E if not e in ans]

    return ans



def getRandomVar(x,y):
    a = np.c_[x,y]
    counter = 0
    lst = a.tolist()
    lst = [tuple(i) for i in lst]
    items = set(lst)
    dic = {}
    for item in items:
        dic[item] = counter
        counter = counter +1
    ans = []
    for item in lst:
        ans.append(dic[item])
    return ans
'''
check if: If (𝑂 ⊥⊥ 𝑅𝐸 = 1|𝐸) and (𝑂 ⊥⊥ 𝑅𝐸 = 1|𝐸,𝑇 )
'''
def checkSelectionBias(x,y,e):
    R = []
    for i in e:
        if i == -1:
            R.append(0)
        else:
            R.append(1)
    Y = [int(i) for i in y]
    E = [int(i) for i in e]
    val = drv.information_mutual_conditional(Y,R,E)
    if val < EPSILON:
        X = [int(i) for i in x]
        ET = getRandomVar(E,X)
        val = drv.information_mutual_conditional(Y,R,ET)
        if val < EPSILON:
            return False
    return True




def getR(x,y,E):
    Re = 0
    for e1, e2 in zip(E, E[1:]):
        varE1 = E[e1].tolist()
        varE2 = E[e2].tolist()
        varE1 = [int(ee) for ee in varE1]
        varE2 = [int(ee) for ee in varE2]
        Re = Re + drv.information_mutual(varE1, varE2)
    Re = Re / (len(E) * len(E))
    return Re

def getD(x,y,E):
    X = [int(ee) for ee in x]
    Y = [int(ee) for ee in y]
    De = 0
    for e in E:
        varE = E[e].tolist()
        varE = [int(ee) for ee in varE]
        valE = drv.information_mutual_conditional(X, Y, varE)
        De += valE

    De = De / len(E)
    return De


def responsibility(x,y,E,ei):
    x = [int(ee) for ee in x]
    y = [int(ee) for ee in y]

    columns = list(E.columns)
    varE = E[columns[0]].tolist()
    varE = [int(ee) for ee in varE]
    for e in columns[1:]:
        varEE = E[e].tolist()
        varEE = [int(ee) for ee in varEE]
        varE = getRandomVar(varE, varEE)
    valE = drv.information_mutual_conditional(x, y, varE)

    valE_ = cmi(E, columns, ei, x, y)

    up = (valE_ - valE)
    s = 0

    Columns = list(E.columns)
    for e in Columns:
        columns = list(E.columns)
        vale = cmi(E, columns, e, x, y)
        s = s + (vale -valE)
    down = s
    #print(up,down,up/down)
    return up/down



def cmi(E, columns, ei, x, y):
    columns.remove(ei)
    varE = E[columns[0]].tolist()
    varE = [int(ee) for ee in varE]
    for e in columns[1:]:
        varEE = E[e].tolist()
        varEE = [int(ee) for ee in varEE]
        varE = getRandomVar(varE, varEE)
    valE_ = drv.information_mutual_conditional(x, y, varE)
    return valE_





def isBiased(X,Y,E):
    """
    Check if the attribute E is biased w.r.t. X,Y
    Pr[R=1|X,Y] = Pr[R = 1]
    Pr[R=1|X,Y] = Pr[R=1|X]
    """
    R = []
    for i in E:
        if i == -1:
            R.append(0)
        else:
            R.append(1)
    PR1 = R.count(1)/len(R)
    dicXY = {}
    dicX = {}
    n = len(X)
    for i in range(n):
        key = str(X[i])+","+str(Y[i])
        keyx = str(X[i])
        if key in dicXY:
            dicXY[key] = dicXY[key] + R[i]
        else:
            dicXY[key] = R[i]
        if keyx in dicX:
            dicX[keyx] = dicX[keyx] + R[i]
        else:
            dicX[keyx] = R[i]
    # check if condition 1 holds
    flag = False
    for key in dicXY:
        val = dicXY[key]
        if not val/n == PR1:
            flag = True
            continue
    if not flag:
        return False
    # check if condition 2 holds
    for key in dicXY:
        val = dicXY[key]
        keyx = key.split(",")[0]
        if not val/n == dicX[keyx]/n:
            return True
    return False




if __name__ == '__main__':
    X = [1,0,1,1,0,1]
    Y = [1,0,1,1,0,1]
    E = [1,-1,0,-1,0,1]
    print(isBiased(X,Y,E))


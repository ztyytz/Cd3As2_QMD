import csv
import numpy as np
from numba import njit, complex64
import matplotlib.pyplot as plt
import time

plt.rcParams['xtick.direction'] = 'in'#
plt.rcParams['ytick.direction'] = 'in'#



fname = "80nm_data/Peirel_2T_80nm.csv"
num_band = 16
theta = np.pi/2
Bx  = 2

def set_tb_disp_kmesh(n_k: int, size_of_BZ: float) -> tuple:

    # n_k is the total num of intervals along the linecut. An odd n_k means (0,0) is not included.

    # num_sec = len(high_symm_pnts)
    num_kpt = n_k
    length = 0

    # klen = np.zeros((num_sec), float)
    kline = np.zeros((num_kpt+1), float)
    kmesh = np.zeros((num_kpt+1, num_kpt+1, 2), float) # Extend to 2D mesh
    step = 2*size_of_BZ/num_kpt
    kline[0] = -size_of_BZ
    kmesh[0, 0] = np.array([-size_of_BZ, -size_of_BZ])
    x_incr = np.array([step, 0])
    y_incr = np.array([0, step])
    
    # for key in high_symm_pnts:
    #     ksec.append(high_symm_pnts[key])

    
    for j in range(num_kpt):
        kmesh[0, j+1] = kmesh[0, j] + y_incr
    
    for i in range(num_kpt):
        kmesh[i+1, 0] = kmesh[i, 0] + x_incr
        kline[i+1] = kline[i] + step
        for j in range(num_kpt):
            kmesh[i+1, j+1] = kmesh[i+1, j] + y_incr

    '''
    for i in range(num_sec-1):
        vec = ksec[i+1]-ksec[i]
        length = np.sqrt(np.dot(vec, vec))
        klen[i+1] = klen[i]+length

        for ikpt in range(n_k):
            kline[ikpt+i*n_k] = klen[i]+ikpt*length/n_k
            kmesh[ikpt+i*n_k] = ksec[i]+ikpt*vec/n_k
    kline[num_kpt] = kline[(num_sec-2)*n_k]+length
    kmesh[num_kpt] = ksec[-1]
    '''
    return (kline, kmesh)

'''
FOR BACKUP

def set_tb_disp_kmesh(n_k: int, high_symm_pnts: dict) -> tuple:

    num_sec = len(high_symm_pnts)
    num_kpt = n_k*(num_sec-1)
    length = 0

    klen = np.zeros((num_sec), float)
    kline = np.zeros((num_kpt+1), float)
    kmesh = np.zeros((num_kpt+1, num_kpt+1, 2), float) # Extend to 2D mesh
    ksec = []
    
    for key in high_symm_pnts:
        ksec.append(high_symm_pnts[key])

    for i in range(num_sec-1):
        vec = ksec[i+1]-ksec[i]
        length = np.sqrt(np.dot(vec, vec))
        klen[i+1] = klen[i]+length

        for ikpt in range(n_k):
            kline[ikpt+i*n_k] = klen[i]+ikpt*length/n_k
            kmesh[ikpt+i*n_k] = ksec[i]+ikpt*vec/n_k
    kline[num_kpt] = kline[(num_sec-2)*n_k]+length
    kmesh[num_kpt] = ksec[-1]

    return (kline, kmesh)
'''

def rot_mat(theta):

    rt_mtrx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return rt_mtrx


with open(fname, "r") as file:
    reader = csv.reader(file)
    data = []
    for row in reader:
        line_data = []
        for x in row:
            s = x.replace("*I", "j")
            line_data.append(s.replace("*^", "e"))
        data.append(line_data)


print("finish loading data of coeff of kp.")
largeComplexArray = np.array(data, dtype=complex)
largeComplexArray = largeComplexArray.reshape(7, 200, 200)

term_const = largeComplexArray[0, :, :]
term_k1    = largeComplexArray[1, :, :]
term_k2    = largeComplexArray[2, :, :]
term_k1k1  = largeComplexArray[3, :, :]
term_k1k2  = largeComplexArray[4, :, :]
term_k2k1  = largeComplexArray[5, :, :]
term_k2k2  = largeComplexArray[6, :, :]

# term_k1k1k1 = largeComplexArray[7,:,:]
# term_k1k1k2 = largeComplexArray[8,:,:]
# term_k1k2k1 = largeComplexArray[9,:,:]
# term_k1k2k2 = largeComplexArray[10,:,:]
# term_k2k1k1 = largeComplexArray[11,:,:]
# term_k2k1k2 = largeComplexArray[12,:,:]
# term_k2k2k1 = largeComplexArray[13,:,:]
# term_k2k2k2 = largeComplexArray[14,:,:]

@njit
def cal_hamk(k1, k2):

    hamk =  term_const+k1*term_k1+k2*term_k2+k1*k2*term_k1k2+k1*k2*term_k2k1+k1**2*term_k1k1+term_k2k2*k2**2 # \
           # +term_k1k1k1*k1*k1*k1+term_k1k1k2*k1*k1*k2+term_k1k2k1*k1*k2*k1+term_k1k2k2*k1*k2*k2 \
           # +term_k2k1k1*k2*k1*k1+term_k2k1k2*k2*k1*k2+term_k2k2k1*k2*k2*k1+term_k2k2k2*k2*k2*k2

    return hamk

@njit
def make_vkx(k1, k2):

    vkx = term_k1+term_k1k2*k2+term_k2k1*k2+term_k1k1*2*k1 # \
        # + term_k1k1k1*3*k1*k1+term_k1k1k2*2*k1*k2+term_k1k2k1*2*k1*k2+term_k1k2k2*k2*k2\
        # + term_k2k1k1*2*k1*k2+term_k2k1k2*k2*k2+term_k2k2k1*k2*k2

    return vkx[:num_band, :num_band]


@njit
def make_vky(k1, k2):

    vky = term_k2+term_k1k2*k1+term_k2k1*k1+term_k2k2*2*k2 # \
        # + term_k1k1k2*k1*k1+term_k1k2k1*k1*k1+term_k1k2k2*k1*k2*2 \
        # +term_k2k1k1*k1*k1+term_k2k1k2*k1*k2*2+term_k2k2k1*k2*k1*2+term_k2k2k2*k2*k2*3

    return vky[:num_band, :num_band]


@njit
def make_Zeeman(B, theta=theta):
    
    muB = 5.7884*1e-5 # [eV/Tesla]
    g1  = 11*muB
    g2  = 0.7*muB  
    gz1 = 10*muB
    gz2 = -5*muB

    Bx = B*np.cos(theta)
    By = B*np.sin(theta)
    B1 = (1/(3*np.sqrt(2))+np.sqrt(2)/3)*Bx+1/np.sqrt(3)*By
    B2 = (-1/(3*np.sqrt(2))-np.sqrt(2)/3)*Bx+1/np.sqrt(3)*By
    B3 = -1/np.sqrt(3)*By

    print("B1 B2 B3 in [T]:", B1, B2, B3)
    # assert np.allclose(B1+B2+2*B3, 0) == True
    # assert np.allclose(np.sqrt(B1**2+B2**2+B3**2), B) == True 

    zeeman = np.zeros((200, 200), dtype=complex64)
    for i in range(50):
        zeeman[4*i:4*i+4, 4*i:4*i+4] = np.array([[gz1*B3, g1*(B1-1j*B2), 0, 0],
                                                 [g1*(B1+1j*B2), -gz1*B3, 0, 0],
                                                 [0, 0, gz2*B3, g2*(B1+1j*B2)],
                                                 [0, 0, g2*(B1-1j*B2), -gz2*B3]])

    return zeeman

zeeman = make_Zeeman(Bx)

@njit
def cal_hamk_mini(hamk):

    hamk_mini = hamk[:num_band, :num_band]

    return hamk_mini

@njit
def make_hamk(kx, ky):
    
    hamk = cal_hamk(kx, ky)
    # hamk = zeeman
    hamk = hamk+zeeman
    hamk_mini = cal_hamk_mini(hamk)

    return hamk_mini


def tolerance(value, target, tol):
    if abs(value-target)<tol:
        return 1
    return 0

def QMElement(n: int, m: int, i, j, dir, step, DeltaE):
    dir = int(dir=='x')
    k1, k2 = kmesh[i, j]
    shift = [k1-step*dir, k2-step*(1-dir)]
    hamk_shift = make_hamk(shift[0], shift[1])
    element = np.dot(eig_funs[i, j, :, n].conj(), np.dot(hamk_shift, eig_funs[i, j, :, m]))/(step*DeltaE)
    return element

def QMIntegrand(n: int, m: int, i, j, step):
    # k1, k2 = kmesh[i, j]
    if m==n:
        return 0
    vx = (eig_vals[i, j, n] - eig_vals[i-1, j, n])/step
    vy = (eig_vals[i, j, n] - eig_vals[i, j-1, n])/step
    DeltaE = eig_vals[i, j, n]-eig_vals[i, j, m]
    value = (vy*QMElement(n, m, i, j, 'x', step, DeltaE)*QMElement(m, n, i, j, 'x', step, DeltaE) - vx*QMElement(n, m, i, j, 'y', step, DeltaE)*QMElement(m, n, i, j, 'x', step, DeltaE))/DeltaE
    return value

def QMDIntegral(QMDipole, band, ktemp, step, miu=0):
    if ktemp<=0:
        return 0
    value = 0.0
    for i in range(n_k):
        for j in range(n_k):
            FDdistdiff = np.exp((eig_vals[i, j, band] - miu)/ktemp)/(ktemp*(1 + np.exp((eig_vals[i, j, band] - miu)/ktemp))**2)
            value = value + FDdistdiff * QMDipole[i, j] * step**2
    return value

if __name__ == "__main__":
    '''
    # Test mode
    print("Test mode")
    kline, kmesh = set_tb_disp_kmesh(4, 1)
    print(kmesh[0,:])
    print(kmesh[:,0])
    '''

    
    # lattice constant
    a = 12.9077420000000007
    c = 25.9834260000000015
    a1 = np.array([a,  0,  0])
    a2 = np.array([0,  a,  0])
    a3 = np.array([0,  0,  c])
    vol = np.dot(a1, np.cross(a2, a3))
    b1 = 2*np.pi*(np.cross(a2, a3))/vol
    b2 = 2*np.pi*(np.cross(a3, a1))/vol
    b3 = 2*np.pi*(np.cross(a1, a2))/vol
    # high symmetry point
    Gamma = np.array([0, 0])
    X = 1/2*b3
    Z = 1/2*b1+1/2*b2-1/2*b3
    # b1 b2 after rotation 
    # b1 = b1[:2]@rot_mat(theta)
    # b2 = b2[:2]@rot_mat(theta)
    b1 = b1[:2]
    b2 = b2[:2]
    print("b1 b2", b1, b2)
    X_n = -1/10*b2
    X_p = 1/10*b2
    print("Xn, Xp: ", X_n, X_p)

    n_k = 200
    # high_symm_points = {'-X':X_n, 'Gamma':Gamma, 'X':X_p}
    size_of_BZ = np.linalg.norm(b1)/10
    step = 2*size_of_BZ/n_k
    kline, kmesh = set_tb_disp_kmesh(n_k, size_of_BZ)
    eig_vals = []
    eig_funs = []

    '''
    # Test mode
    print("Test mode")
    print(kmesh[0,:])
    print(kmesh[:,0])
    '''
    count = 0
    for kcut in kmesh:
        for kpnt in kcut:
            k1, k2 = kpnt
            hamk = make_hamk(k1, k2)
            eigv, eigf = np.linalg.eigh(hamk)
            eig_vals.append(eigv[np.argsort(eigv)])
            for i in range(np.shape(eigf)[0]):
                eigf[:][i] = eigf[:][i]/np.linalg.norm(eigf[:][i])      # Normalizing column vectors
            eig_funs.append(eigf[np.argsort(eigv)])
        # print("Finishing one column eigsys")
            # print("Mesh" + str(count+1))
            # print(np.dot(hamk, eigf[:, 0])-eigv[0]*eigf[:, 0])
            # print(np.dot(hamk, eig_funs[-1][:, 0])-eig_vals[-1][0]*eig_funs[-1][:, 0])
            # print(eig_funs[-1][:, 0])
            # print(eig_funs[-1])
            # print(eigv)
            # print(eig_vals)
            # count = count+1
    # eig_vals = np.array(eig_vals)

    # Test mode
    # print("Test mode")
    # print(eig_vals)
    # print(np.shape(eig_funs))
    # test = eig_funs[2]
    # print(test)
    # print(".........................")
    # for i in range(4):
        # tmp.append(np.dot(make_hamk(kmesh[0, 0, 0], kmesh[0, 0, 1]), eig_funs[0][:, i])-eig_vals[0, i]*eig_funs[0][:, i])
        # print("eig_funs - "+str(i+1)+": ")
        # print(eig_funs[0][:, i])
    # print(tmp)

    
    eig_funs = np.array(eig_funs).reshape(n_k+1, n_k+1, num_band, num_band)
    eig_vals = np.array(eig_vals).reshape(n_k+1, n_k+1, num_band)
    # The first two indices correspond to x and y coordinates, specifying a point in BZ
    # To acquire wavefunction, use vertical vector: eig_funs[..., ..., :, i]

    
    # Checking reshape operations
    '''
    tmp = []
    count = 0
    for i in range(n_k+1):
        for j in range(n_k+1):
            hamk = make_hamk(kmesh[i, j, 0], kmesh[i, j, 1])
            for k in range(num_band):
                tmp.append(np.dot(hamk, eig_funs[i, j, :, k])-eig_vals[i, j, k]*eig_funs[i, j, :, k])
    tmp = np.array(tmp)
    print(np.linalg.norm(tmp))
    '''

    # Checking normalization
    '''
    tmp = []
    for i in range(n_k+1):
        for j in range(n_k+1):
            for k in range(num_band):
                tmp.append(np.linalg.norm(eig_funs[i, j, :, k]))
    print(np.linalg.norm(np.array(tmp)-np.ones(np.shape(tmp)[0])))
    '''

    # Calculate QMD
    # '''
    qmdlist = []
    for n in range(num_band):
        print("Calculating QM distribution for band # "+str(n+1))
        qmd = []
        for i in range(n_k):
            for j in range(n_k):
                qmd_pnt = 0
                for m in range(num_band):
                    qmd_pnt = qmd_pnt + QMIntegrand(n, m, i, j, step)
                qmd.append([i, j, qmd_pnt])
            if (i+1)%10 == 0:
                print("QM calculation progress " + str(i+1) + " out of " + str(n_k))
        qmd = np.array(qmd)
        qmdlist.append(qmd)
    # qmdsum = qmdlist[0]
    # for i in range(len(qmdlist)-1):
    #     qmdsum = qmdsum + qmdlist[i+1]
    QMDipole = []
    for i in range(num_band):
        QMDipole.append(np.real(qmdlist[i][:, 2]).reshape(n_k, n_k)/1e6)
    # QMDipole = np.real(qmdsum[:, 2]).reshape(n_k, n_k)/1e6
    # '''

    '''
    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    QMDipole = np.real(qmd[:, 2]).reshape(n_k, n_k)
    cax = ax.matshow(QMDipole, interpolation='nearest', cmap='RdBu')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    plt.xlabel(r'$k_x$',fontsize=24)
    plt.ylabel(r'$k_y$',fontsize=24)
    cb1 = plt.colorbar(cax)
    #cb1.set_ticks([])
    cb1.set_label(r'$gxx*vy-gxy*vx$',fontsize=24)
    cb1.update_ticks()
    # im = ax.imshow(QMDipole)
    # fig.colorbar(im, ax=ax)
    plt.show()
    # '''
    
    # Band structure + QM distribution plot
    '''
    fig = plt.figure(figsize = (10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    x_axis = kmesh[: ,: ,0]
    y_axis = kmesh[: ,: ,1]
    z_axis = eig_vals[:, :, n]
    ax1.plot_surface(x_axis, y_axis, z_axis)
    # ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 2, 2)
    cax = ax3.matshow(QMDipole, interpolation='nearest', cmap='RdBu', vmax = 4, vmin = -4)
    # QMDipole_rescaled = np.sqrt(abs(QMDipole[:, :])) * np.sign(QMDipole[:, :])
    # cax = ax3.matshow(QMDipole_rescaled, interpolation='nearest', cmap='RdBu', vmax = 2, vmin = -2)
    bound = np.round(size_of_BZ, decimals=3)
    ax3.xaxis.set_ticks([0, n_k/2, n_k])
    ax3.xaxis.set_ticklabels([str(-bound), '0', str(bound)])
    ax3.yaxis.set_ticks([0, n_k/2, n_k])
    ax3.yaxis.set_ticklabels([str(-bound), '0', str(bound)])
    plt.xlabel(r'$k_x$',fontsize=12)
    plt.ylabel(r'$k_y$',fontsize=12)
    cb1 = plt.colorbar(cax)
    #cb1.set_ticks([])
    cb1.set_label(r'Quantum Metric',fontsize=12)
    cb1.update_ticks()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    # im = ax.imshow(QMDipole)
    # fig.colorbar(im, ax=ax)
    plt.show()
    '''

    # QM distribution + QMD-chemical potential plot
    # '''
    fig = plt.figure(figsize = (10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    potential = np.linspace(-0.1, 0.2, 60)
    QMD = []
    T = 0.01
    for mu in potential:
        QMDtemp = 0
        for n in range(num_band):
            QMDtemp = QMDtemp + QMDIntegral(QMDipole[n], n, T, step, mu)
        QMD.append(QMDtemp)
    ax1.set_xlabel("Chemical Potential")
    ax1.set_ylabel("Full QM dipole")
    ax1.plot(potential, QMD, **{'ls':'-', 'color':'#4A90E2', 'lw':1})
    # ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 2, 2)
    # QMDipole = np.real(qmd[:, 2]).reshape(n_k, n_k)/1e6
    cax = ax3.matshow(QMDipole[int(num_band/2)], interpolation='nearest', cmap='RdBu', vmax = 4, vmin = -4)
    # QMDipole_rescaled = np.sqrt(abs(QMDipole[:, :])) * np.sign(QMDipole[:, :])
    # cax = ax3.matshow(QMDipole_rescaled, interpolation='nearest', cmap='RdBu', vmax = 2, vmin = -2)
    bound = np.round(size_of_BZ, decimals=3)
    ax3.xaxis.set_ticks([0, n_k/2, n_k])
    ax3.xaxis.set_ticklabels([str(-bound), '0', str(bound)])
    ax3.yaxis.set_ticks([0, n_k/2, n_k])
    ax3.yaxis.set_ticklabels([str(-bound), '0', str(bound)])
    plt.xlabel(r'$k_x$',fontsize=12)
    plt.ylabel(r'$k_y$',fontsize=12)
    cb1 = plt.colorbar(cax)
    #cb1.set_ticks([])
    cb1.set_label(r'Quantum Metric',fontsize=12)
    cb1.update_ticks()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    # fig.savefig(str(Bx)+'T_'+str(n+1)+'in'+str(num_band)+'_T='+str(T)+'.png')
    fig.savefig('80nm_fig/'+str(Bx)+'T_All'+str(num_band)+'Band_T='+str(T)+'.png')
    # im = ax.imshow(QMDipole)
    # fig.colorbar(im, ax=ax)
    plt.show()
    # '''

    # Multiple 3D band plots
    '''
    fig = plt.figure(figsize=(12, 6))
    # ax = fig.add_subplot(projection='3d')
    # fig, axs = plt.subplots(2, 3, figsize=(6, 6))
    x_axis = kmesh[: ,: ,0]
    y_axis = kmesh[: ,: ,1]
    for i in range(2):
        for j in range(4):
            ax = fig.add_subplot(2, 4, 4*i+j+1, projection='3d')
            z_axis = eig_vals[:, :, 4*i+j].reshape(n_k+1, n_k+1)
            ax.plot_surface(x_axis, y_axis, z_axis)
    # ax.set_aspect('equal')
    
    # z_axis = eig_vals[:, 2].reshape(n_k+1, n_k+1)
    # ax.plot_surface(x_axis, y_axis, z_axis)
    fig.tight_layout()
    plt.show()

    '''

    # Fermi contour plotting
    '''
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), layout='constrained')
    en = eig_vals[:, 2].reshape(n_k+1, n_k+1)
    # plt.contour(kmesh[:, :, 0], kmesh[:, :, 1], tolerance(eig_vals[:, 2].reshape(n_k+1, n_k+1)[:, :],) ,20)
    val = np.zeros((n_k+1, n_k+1))
    for i in range(n_k+1):
        for j in range(n_k+1):
            val[i, j] = tolerance(en[i, j], 0, 0.0003)
    im = ax.imshow(val)
    fig.colorbar(im, ax=ax)
    plt.show()
    '''
    
    # Original codes
    '''
    for kpnt in kmesh:
        k1, k2 = kpnt
        hamk = make_hamk(k1, k2)
        eig, _ = np.linalg.eigh(hamk)
        eig_vals.append(eig)
    eig_vals = np.array(eig_vals)
    '''
    
    '''
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), layout='constrained')
    for i in range(num_band):
        ax.plot(kline, eig_vals[:, :,  i], **{'ls':'-', 'color':'#4A90E2', 'lw':1})
    ax.set_xlim(0, kline[-1])
    ax.set_ylim([-0.025, 0.025])
    knorm = np.round(np.sqrt(X_n[0]**2+X_n[1]**2),decimals=3)
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k]])
    ax.set_xticklabels(["-"+str(knorm), "$k_y$ (ang$^{-1}$)", str(knorm)])
    ax.axvline(x=kline[n_k], color="grey", lw=0.8, ls='--')
    ax.set_ylabel("Energy (eV)")
    ax.set_title("[112] 20 nm "+str(Bx)+" T")
    #plt.savefig("112_r2.png", dpi=500)
    #np.save("kline.npy", kline)
    #np.save("band_112_"+str(Bx)+"T.npy", eig_vals)
    plt.show()
    '''
    
    




import os
import glob
import numpy as np
import matplotlib.pyplot as plt


label_q = "quant_"
label_c = "class_"
directory = './tmp/'
directory = '.'


pol_data1_q = np.load(glob.glob(os.path.join(directory, "pol_data1_" + label_q + "*.npy"))[0])
pol_data2_q = np.load(glob.glob(os.path.join(directory, "pol_data2_" + label_q + "*.npy"))[0])
pop_data_cond_q = np.load(glob.glob(os.path.join(directory, "pop_data_cond_" + label_q + "*.npy"))[0])
pop_data_val_q = np.load(glob.glob(os.path.join(directory, "pop_data_val_" + label_q + "*.npy"))[0])

pol_data1_c = np.load(glob.glob(os.path.join(directory, "pol_data1_" + label_c + "*.npy"))[0])
pol_data2_c = np.load(glob.glob(os.path.join(directory, "pol_data2_" + label_c + "*.npy"))[0])
pop_data_cond_c = np.load(glob.glob(os.path.join(directory, "pop_data_cond_" + label_c + "*.npy"))[0])
pop_data_val_c = np.load(glob.glob(os.path.join(directory, "pop_data_val_" + label_c + "*.npy"))[0])

time_c = np.load(glob.glob(os.path.join(directory, "time_" + label_c + "*.npy"))[0])
field_c = np.load(glob.glob(os.path.join(directory, "field_" + label_c + "*.npy"))[0])
time_q = np.load(glob.glob(os.path.join(directory, "time_" + label_q + "*.npy"))[0])
field_q = np.load(glob.glob(os.path.join(directory, "field_" + label_q + "*.npy"))[0])


pad = np.zeros(len(time_q) + 10000)
pad[:len(time_q)] = field_q
E_f_q = np.fft.fft(pad)
plt.plot(np.real(E_f_q))
plt.plot(np.imag(E_f_q))

pad = np.zeros(len(time_c) + 10000)
pad[:len(time_c)] = field_c
E_f_c = np.fft.fft(pad)
plt.plot(np.real(E_f_c))
plt.plot(np.imag(E_f_c))
plt.show()

if pol_data1_c.shape[0] == 1:

    pp_c = pol_data1_c[0, :] + 1j * pol_data2_c[0, :]
    pp_q = pol_data1_q[0, :] + 1j * pol_data2_q[0, :]
    plt.plot(np.real(pp_c))
    plt.plot(np.imag(pp_c))
    plt.plot(np.real(pp_q))
    plt.plot(np.imag(pp_q))
    plt.show()

    plt.plot(pol_data1_c.T)
    plt.plot(pol_data1_q.T)
    plt.show()

    plt.plot(pop_data_cond_c.T)
    plt.plot(pop_data_cond_q.T)
    plt.show()

    plt.plot(pop_data_val_c.T)
    plt.plot(pop_data_val_q.T)
    plt.show()

else:

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].contourf(pol_data1_c, 50)
    ax[1].contourf(pol_data1_q, 50)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].contourf(pol_data2_c, 50)
    ax[1].contourf(pol_data2_q, 50)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].contourf(pop_data_cond_c, 50)
    ax[1].contourf(pop_data_val_q, 50)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].contourf(pop_data_val_c, 50)
    ax[1].contourf(pop_data_cond_q, 50)
    plt.show()

    pp_c = np.sum(pol_data1_c, axis=0) + 1j * np.sum(pol_data2_c, axis=0)
    pp_q = np.sum(pol_data1_q, axis=0) + 1j * np.sum(pol_data2_q, axis=0)

    plt.plot(np.real(pp_c))
    plt.plot(np.imag(pp_c))
    plt.plot(np.real(pp_q), '--')
    plt.plot(np.imag(pp_q), '--')
    plt.show()

    plt.plot(pol_data1_c.T)
    plt.plot(pol_data1_q.T)
    plt.show()

    plt.plot(pop_data_cond_c.T)
    plt.plot(pop_data_val_q.T)
    plt.show()

    plt.plot(pop_data_val_c.T)
    plt.plot(pop_data_cond_q.T)
    plt.show()

pad = np.zeros(len(pp_c) + 10000, dtype=complex)
pad[:len(time_c)] = pp_c
pp_f_c = np.fft.fft(pad)
freqs_c = np.arange(len(pp_f_c)) / ((time_c[2] - time_c[1]) * len(pp_f_c)) * 2 * np.pi
# freqs_c = np.fft.fftfreq(len(pp_f_c), (time_c[2] - time_c[1])) * 2 * np.pi

pad = np.zeros(len(pp_q) + 10000, dtype=complex)
pad[:len(time_q)] = pp_q
pp_f_q = np.fft.fft(pad)
freqs_q = np.arange(len(pp_f_q)) / ((time_q[2] - time_q[1]) * len(pp_f_q)) * 2 * np.pi
# freqs_q = np.fft.fftfreq(len(pp_f_q), (time_q[2] - time_q[1])) * 2 * np.pi





freqs_c = np.fft.fftfreq(len(pp_f_c), (time_c[2] - time_c[1])) * 2 * np.pi
ind_c = np.argsort(freqs_c)
freqs_c = freqs_c[ind_c]
pp_f_c = pp_f_c[ind_c]
E_f_c = E_f_c[ind_c]
freqs1 = freqs_c[freqs_c >= 0]
freqs2 = freqs_c[freqs_c < 0]
lim = min(len(freqs1), len(freqs2))
pp_f_c = pp_f_c[freqs_c >= 0][:lim] + np.conj(pp_f_c[freqs_c < 0])[:lim][::-1]
E_f_c = E_f_c[freqs_c >= 0][:lim]
freqs_c = freqs_c[freqs_c >= 0][:lim]
plt.plot(freqs_c, np.real(pp_f_c / E_f_c))
plt.plot(freqs_c, np.imag(pp_f_c / E_f_c))

freqs_q = np.fft.fftfreq(len(pp_f_q), (time_q[2] - time_q[1])) * 2 * np.pi
ind_q = np.argsort(freqs_q)
freqs_q = freqs_q[ind_q]
pp_f_q = pp_f_q[ind_q]
E_f_q = E_f_q[ind_q]
freqs1 = freqs_q[freqs_q >= 0]
freqs2 = freqs_q[freqs_q < 0]
lim = min(len(freqs1), len(freqs2))
pp_f_q = pp_f_q[freqs_q >= 0][:lim] + np.conj(pp_f_q[freqs_q < 0])[:lim][::-1]
E_f_q = E_f_q[freqs_q >= 0][:lim]
freqs_q = freqs_q[freqs_q >= 0][:lim]
plt.plot(freqs_q, np.real(pp_f_q / E_f_q), '--')
plt.plot(freqs_q, np.imag(pp_f_q / E_f_q), '--')
# plt.xlim([0, 5])
plt.show()

# plt.plot(freqs_c[ind_c], np.real(pp_f_c / E_f_c)[ind_c])
# plt.plot(freqs_c[ind_c], np.imag(pp_f_c / E_f_c)[ind_c])
# plt.plot(freqs_q[ind_q], np.real(pp_f_q / E_f_q)[ind_q], '--')
# plt.plot(freqs_q[ind_q], np.imag(pp_f_q / E_f_q)[ind_q], '--')
# # plt.xlim([0, 5])
# plt.show()

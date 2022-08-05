#include <omp.h>

#include <CL/sycl.hpp>
#include <thread>

#include "Defines.h"

inline void correntess(ComplexType result1, ComplexType result2,
                       ComplexType result3) {
  double re_diff, im_diff;
  int numThreads = 128;
  // #pragma omp parallel
  //   {
  //     int ttid = omp_get_thread_num();
  //     if (ttid == 0) numThreads = omp_get_num_threads();
  //   }
  printf("here are %d threads \n", numThreads);
  if (numThreads <= 64) {
    re_diff = fabs(result1.real() - -264241151.454552);
    im_diff = fabs(result1.imag() - 1321205770.975190);
    re_diff += fabs(result2.real() - -137405397.758745);
    im_diff += fabs(result2.imag() - 961837795.884157);
    re_diff += fabs(result3.real() - -83783779.241634);
    im_diff += fabs(result3.imag() - 754054017.424472);
    printf("%f,%f\n", re_diff, im_diff);
  } else {
    re_diff = fabs(result1.real() - -264241151.200123);
    im_diff = fabs(result1.imag() - 1321205763.246570);
    re_diff += fabs(result2.real() - -137405398.773852);
    im_diff += fabs(result2.imag() - 961837794.726070);
    re_diff += fabs(result3.real() - -83783779.939936);
    im_diff += fabs(result3.imag() - 754054018.099450);
  }
  if (re_diff < 10 && im_diff < 10)
    printf("\n!!!! SUCCESS - !!!! Correctness test passed :-D :-D\n\n");
  else
    printf("\n!!!! FAILURE - Correctness test failed :-( :-(  \n");
}

int main(int argc, char **argv) {
  int number_bands = 0, nvband = 0, ncouls = 0, nodes_per_group = 0;
  int npes = 1;
  if (argc == 1) {
    number_bands = 512;
    nvband = 2;
    ncouls = 32768;
    nodes_per_group = 20;
  } else if (argc == 5) {
    number_bands = atoi(argv[1]);
    nvband = atoi(argv[2]);
    ncouls = atoi(argv[3]);
    nodes_per_group = atoi(argv[4]);
  } else {
    std::cout << "The correct form of input is : " << std::endl;
    std::cout << " ./main.exe <number_bands> <number_valence_bands> "
                 "<number_plane_waves> <nodes_per_mpi_group> "
              << std::endl;
    exit(0);
  }
  int ngpown = ncouls / (nodes_per_group * npes);

  // Constants that will be used later
  const DataType e_lk = 10;
  const DataType dw = 1;
  const DataType to1 = 1e-6;
  const DataType limittwo = pow(0.5, 2);
  const DataType e_n1kq = 6.0;

  // Using time point and system_clock
  time_point<system_clock> start, end, k_start, k_end;
  start = system_clock::now();
  double elapsedKernelTimer;

  // Printing out the params passed.
  std::cout << "Sizeof(ComplexType = " << sizeof(ComplexType) << " bytes"
            << std::endl;
  std::cout << "number_bands = " << number_bands << "\t nvband = " << nvband
            << "\t ncouls = " << ncouls
            << "\t nodes_per_group  = " << nodes_per_group
            << "\t ngpown = " << ngpown << "\t nend = " << nend
            << "\t nstart = " << nstart << std::endl;

  size_t memFootPrint = 0.00;

  // ALLOCATE statements .
  ARRAY1D achtemp(nend - nstart);
  memFootPrint += (nend - nstart) * sizeof(ComplexType);

  ARRAY2D aqsmtemp(number_bands, ncouls);
  ARRAY2D aqsntemp(number_bands, ncouls);
  memFootPrint += 2 * (number_bands * ncouls) * sizeof(ComplexType);

  ARRAY2D I_eps_array(ngpown, ncouls);
  ARRAY2D wtilde_array(ngpown, ncouls);
  memFootPrint += 2 * (ngpown * ncouls) * sizeof(ComplexType);

  ARRAY1D_DataType vcoul(ncouls);
  memFootPrint += ncouls * sizeof(DataType);

  ARRAY1D_int inv_igp_index(ngpown);
  ARRAY1D_int indinv(ncouls + 1);
  memFootPrint += ngpown * sizeof(int);
  memFootPrint += (ncouls + 1) * sizeof(int);

  ARRAY1D_DataType wx_array(nend - nstart);
  memFootPrint += 3 * (nend - nstart) * sizeof(DataType);

  // Print Memory Foot print
  std::cout << "Memory Foot Print = " << memFootPrint / pow(1024, 3) << " GBs"
            << std::endl;

  ComplexType expr(.5, .5);
  for (int i = 0; i < number_bands; i++)
    for (int j = 0; j < ncouls; j++) {
      aqsmtemp(i, j) = expr;
      aqsntemp(i, j) = expr;
    }

  for (int i = 0; i < ngpown; i++)
    for (int j = 0; j < ncouls; j++) {
      I_eps_array(i, j) = expr;
      wtilde_array(i, j) = expr;
    }

  for (int i = 0; i < ncouls; i++) vcoul(i) = 1.0;

  for (int ig = 0; ig < ngpown; ++ig)
    inv_igp_index(ig) = (ig + 1) * ncouls / ngpown;

  for (int ig = 0; ig < ncouls; ++ig) indinv(ig) = ig;
  indinv(ncouls) = ncouls - 1;

  for (int iw = nstart; iw < nend; ++iw) {
    wx_array(iw) = e_lk - e_n1kq + dw * ((iw + 1) - 2);
    if (wx_array(iw) < to1) wx_array(iw) = to1;
  }

  k_start = system_clock::now();
  noflagOCC_solver(number_bands, ngpown, ncouls, inv_igp_index, indinv,
                   wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array,
                   vcoul, achtemp);

  k_end = system_clock::now();
  duration<double> elapsed = k_end - k_start;
  elapsedKernelTimer = elapsed.count();

  // Check for correctness
  // correntess0(achtemp(0));
  // correntess1(achtemp(1));
  // correntess2(achtemp(2));
  correntess(achtemp(0), achtemp(1), achtemp(2));
  printf("\n Final achtemp\n");
  ComplexType_print(achtemp(0));
  ComplexType_print(achtemp(1));
  ComplexType_print(achtemp(2));

  end = system_clock::now();
  elapsed = end - start;

  std::cout << "********** Kernel Time Taken **********= " << elapsedKernelTimer
            << " secs" << std::endl;
  std::cout << "********** Total Time Taken **********= " << elapsed.count()
            << " secs" << std::endl;

  return 0;
}

using namespace sycl;

#define MIN(x, y) ((x) < (y) ? (x) : (y))

void noflagOCC_solver(size_t number_bands, size_t ngpown, size_t ncouls,
                      ARRAY1D_int &inv_igp_index, ARRAY1D_int &indinv,
                      ARRAY1D_DataType &wx_array, ARRAY2D &wtilde_array,
                      ARRAY2D &aqsmtemp, ARRAY2D &aqsntemp,
                      ARRAY2D &I_eps_array, ARRAY1D_DataType &vcoul,
                      ARRAY1D &achtemp) {
  auto root_devices = platform(gpu_selector{}).get_devices();
  vector<queue> queues;
  int _number_bands =
      number_bands % 256 == 0 ? number_bands : (((number_bands << 8) + 1) >> 8);
  int _ngpown = (ngpown & 1) == 0 ? ngpown : ngpown + 1;
  int number_bands_per_tile = _number_bands >> 2;
  constexpr int device_num = 4;

  for (auto root_device : root_devices) {
    auto sub_devices = root_device.create_sub_devices<
        cl::sycl::info::partition_property::partition_by_affinity_domain>(
        cl::sycl::info::partition_affinity_domain::next_partitionable);
    for (auto sub_device : sub_devices) {
      queues.push_back(queue(sub_device));
    }
  }

  ARRAY2D sch(number_bands, ngpown);
  for (int n1 = 0; n1 < number_bands; ++n1) {
    for (int my_igp = 0; my_igp < ngpown; ++my_igp) {
      int igp = indinv(inv_igp_index(my_igp));
      sch(n1, my_igp) = ComplexType_conj(aqsmtemp(n1, igp)) *
                        aqsntemp(n1, igp) * 0.5 * vcoul(igp) *
                        wtilde_array(my_igp, igp);
    }
  }

  achtemp(0) = ComplexType(0.00, 0.00);
  achtemp(1) = ComplexType(0.00, 0.00);
  achtemp(2) = ComplexType(0.00, 0.00);

  DataType *ach_re0[device_num];
  DataType *ach_re1[device_num];
  DataType *ach_re2[device_num];
  DataType *ach_im0[device_num];
  DataType *ach_im1[device_num];
  DataType *ach_im2[device_num];

  for (int id = 0; id < device_num; ++id) {
    auto q = queues[id];
    ach_re0[id] = malloc_shared<DataType>(1, q);
    ach_re1[id] = malloc_shared<DataType>(1, q);
    ach_re2[id] = malloc_shared<DataType>(1, q);
    ach_im0[id] = malloc_shared<DataType>(1, q);
    ach_im1[id] = malloc_shared<DataType>(1, q);
    ach_im2[id] = malloc_shared<DataType>(1, q);
    ach_re0[id][0] = 0.00, ach_re1[id][0] = 0.00, ach_re2[id][0] = 0.00,
    ach_im0[id][0] = 0.00, ach_im1[id][0] = 0.00, ach_im2[id][0] = 0.00;
  }

  DataType *gpu_wx_array[device_num];
  ComplexType *gpu_wtilde_array[device_num];
  ComplexType *gpu_I_eps_array[device_num];
  ComplexType *gpu_sch[device_num];

  for (int id = 0; id < device_num; ++id) {
    auto q = queues[id];
    gpu_wx_array[id] = malloc_device<DataType>(nend - nstart, q);
    gpu_wtilde_array[id] = malloc_device<ComplexType>(ngpown * ncouls, q);
    gpu_I_eps_array[id] = malloc_device<ComplexType>(ngpown * ncouls, q);
    gpu_sch[id] = malloc_device<ComplexType>(number_bands_per_tile * ngpown, q);

    q.memcpy(gpu_wx_array[id], wx_array.dptr,
             (nend - nstart) * sizeof(DataType));
    q.memcpy(gpu_wtilde_array[id], wtilde_array.dptr,
             ngpown * ncouls * sizeof(ComplexType));
    q.memcpy(gpu_I_eps_array[id], I_eps_array.dptr,
             ngpown * ncouls * sizeof(ComplexType));
    if (number_bands_per_tile * id < number_bands) {
      int count = ngpown *
                  MIN(number_bands_per_tile,
                      number_bands - number_bands_per_tile * id) *
                  sizeof(ComplexType);
      q.memcpy(gpu_sch[id], sch.dptr + number_bands_per_tile * ngpown * id,
               count);
    }
  }

  for (int id = 0; id < device_num; ++id) {
    queues[id].wait();
  }

  for (int id = 0; id < device_num; ++id) {
    int offset = number_bands_per_tile * id;
    auto q = queues[id];

    q.submit([&](handler &h) {
      auto _ach_re0 = ach_re0[id];
      auto _ach_re1 = ach_re1[id];
      auto _ach_re2 = ach_re2[id];
      auto _ach_im0 = ach_im0[id];
      auto _ach_im1 = ach_im1[id];
      auto _ach_im2 = ach_im2[id];
      auto _gpu_sch = gpu_sch[id];
      auto _gpu_wtilde_array = gpu_wtilde_array[id];
      auto _gpu_wx_array = gpu_wx_array[id];
      auto _gpu_I_eps_array = gpu_I_eps_array[id];

      h.parallel_for(
          nd_range<2>(range<2>(number_bands_per_tile, _ngpown),
                      range<2>(64, 2)),
          reduction(_ach_re0, sycl::plus<>()),
          reduction(_ach_re1, sycl::plus<>()),
          reduction(_ach_re2, sycl::plus<>()),
          reduction(_ach_im0, sycl::plus<>()),
          reduction(_ach_im1, sycl::plus<>()),
          reduction(_ach_im2, sycl::plus<>()),
          [=](nd_item<2> item, auto &_ach_re0, auto &_ach_re1, auto &_ach_re2,
              auto &_ach_im0, auto &_ach_im1,
              auto &_ach_im2) [[intel::reqd_sub_group_size(32)]] {
            int n1 = item.get_global_id(0);
            int my_igp = item.get_global_id(1);

            if (n1 + offset >= number_bands || my_igp >= ngpown) return;

            ComplexType sch_store1 = _gpu_sch[n1 * ngpown + my_igp];

            for (int ig = 0; ig < ncouls; ++ig) {
              auto wdiff =
                  _gpu_wx_array[0] - _gpu_wtilde_array[my_igp * ncouls + ig];
              auto delw = ComplexType_conj(wdiff) *
                          (1 / (wdiff * ComplexType_conj(wdiff)).real());
              auto sch_array =
                  delw * _gpu_I_eps_array[my_igp * ncouls + ig] * sch_store1;
              _ach_re0 += (sch_array).real();
              _ach_im0 += (sch_array).imag();

              wdiff =
                  _gpu_wx_array[1] - _gpu_wtilde_array[my_igp * ncouls + ig];
              delw = ComplexType_conj(wdiff) *
                     (1 / (wdiff * ComplexType_conj(wdiff)).real());
              sch_array =
                  delw * _gpu_I_eps_array[my_igp * ncouls + ig] * sch_store1;
              _ach_re1 += (sch_array).real();
              _ach_im1 += (sch_array).imag();

              wdiff =
                  _gpu_wx_array[2] - _gpu_wtilde_array[my_igp * ncouls + ig];
              delw = ComplexType_conj(wdiff) *
                     (1 / (wdiff * ComplexType_conj(wdiff)).real());
              sch_array =
                  delw * _gpu_I_eps_array[my_igp * ncouls + ig] * sch_store1;
              _ach_re2 += (sch_array).real();
              _ach_im2 += (sch_array).imag();
            }
          });
    });
  }

  for (int id = 0; id < device_num; ++id) {
    queues[id].wait();
  }

  for (int id = 0; id < device_num; ++id) {
    auto q = queues[id];
    achtemp(0) += ComplexType(ach_re0[id][0], ach_im0[id][0]);
    achtemp(1) += ComplexType(ach_re1[id][0], ach_im1[id][0]);
    achtemp(2) += ComplexType(ach_re2[id][0], ach_im2[id][0]);
    sycl::free(gpu_wx_array[id], q);
    sycl::free(gpu_wtilde_array[id], q);
    sycl::free(gpu_I_eps_array[id], q);
    sycl::free(gpu_sch[id], q);
    sycl::free(ach_im0[id], q);
    sycl::free(ach_im1[id], q);
    sycl::free(ach_im2[id], q);
    sycl::free(ach_re0[id], q);
    sycl::free(ach_re1[id], q);
    sycl::free(ach_re2[id], q);
  }
}

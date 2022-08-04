#include <omp.h>

#include <CL/sycl.hpp>

#include "Defines.h"

inline void correntess(ComplexType result1, ComplexType result2,
                       ComplexType result3) {
  double re_diff, im_diff;
  int numThreads;
#pragma omp parallel
  {
    int ttid = omp_get_thread_num();
    if (ttid == 0) numThreads = omp_get_num_threads();
  }
  printf("here are %d threads \n", numThreads);
  // if (numThreads <= 64) {
  //   re_diff = fabs(result1.real() - -264241151.454552);
  //   im_diff = fabs(result1.imag() - 1321205770.975190);
  //   re_diff += fabs(result2.real() - -137405397.758745);
  //   im_diff += fabs(result2.imag() - 961837795.884157);
  //   re_diff += fabs(result3.real() - -83783779.241634);
  //   im_diff += fabs(result3.imag() - 754054017.424472);
  //   printf("%f,%f\n", re_diff, im_diff);
  // } else {
  re_diff = fabs(result1.real() - -264241151.200123);
  im_diff = fabs(result1.imag() - 1321205763.246570);
  re_diff += fabs(result2.real() - -137405398.773852);
  im_diff += fabs(result2.imag() - 961837794.726070);
  re_diff += fabs(result3.real() - -83783779.939936);
  im_diff += fabs(result3.imag() - 754054018.099450);
  // }
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
  cout << "Memory Foot Print = " << memFootPrint / pow(1024, 3) << " GBs"
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

  cout << "********** Kernel Time Taken **********= " << elapsedKernelTimer
       << " secs" << std::endl;
  cout << "********** Total Time Taken **********= " << elapsed.count()
       << " secs" << std::endl;

  return 0;
}

using namespace sycl;

void noflagOCC_solver(size_t number_bands, size_t ngpown, size_t ncouls,
                      ARRAY1D_int &inv_igp_index, ARRAY1D_int &indinv,
                      ARRAY1D_DataType &wx_array, ARRAY2D &wtilde_array,
                      ARRAY2D &aqsmtemp, ARRAY2D &aqsntemp,
                      ARRAY2D &I_eps_array, ARRAY1D_DataType &vcoul,
                      ARRAY1D &achtemp) {
  gpu_selector device_selector;
  queue q(device_selector);
  std::cout << "Device: " << q.get_device().get_info<info::device::name>()
            << std::endl;

  auto ctx = q.get_context();
  int *gpu_inv_igp_index = malloc_device<int>(ngpown, q);
  int *gpu_indinv = malloc_device<int>(ngpown, q);
  DataType *gpu_vcoul = malloc_device<DataType>(ncouls, q);
  DataType *gpu_wx_array = malloc_device<DataType>(nend - nstart, q);
  ComplexType *gpu_wtilde_array =
      malloc_device<ComplexType>(ngpown * ncouls, q);
  ComplexType *gpu_I_eps_array = malloc_device<ComplexType>(ngpown * ncouls, q);
  ComplexType *gpu_aqsntemp =
      malloc_device<ComplexType>(number_bands * ncouls, q);
  ComplexType *gpu_aqsmtemp =
      malloc_device<ComplexType>(number_bands * ncouls, q);
  ComplexType *sch = malloc_device<ComplexType>(number_bands * ncouls, q);

  time_point<system_clock> start, end;

  start = system_clock::now();
  q.memcpy(gpu_inv_igp_index, inv_igp_index.dptr, ngpown * sizeof(int)).wait();
  q.memcpy(gpu_indinv, indinv.dptr, ngpown * sizeof(int)).wait();
  q.memcpy(gpu_vcoul, vcoul.dptr, ngpown * sizeof(DataType)).wait();
  q.memcpy(gpu_wx_array, wx_array.dptr, (nend - nstart) * sizeof(DataType))
      .wait();
  q.memcpy(gpu_wtilde_array, wtilde_array.dptr,
           ngpown * ncouls * sizeof(ComplexType))
      .wait();
  q.memcpy(gpu_I_eps_array, I_eps_array.dptr,
           ngpown * ncouls * sizeof(ComplexType))
      .wait();
  q.memcpy(gpu_aqsntemp, aqsntemp.dptr,
           number_bands * ncouls * sizeof(ComplexType))
      .wait();
  q.memcpy(gpu_aqsmtemp, aqsmtemp.dptr,
           number_bands * ncouls * sizeof(ComplexType))
      .wait();
  end = system_clock::now();
  duration<double> elapsed = end - start;
  double elapsedKernelTimer = elapsed.count();
  cout << "********** Copy Time Taken **********= " << elapsed.count()
       << " secs" << std::endl;

  start = system_clock::now();
  q.submit([&](handler &h) {
     h.parallel_for(range<2>(number_bands, ngpown), [=](item<2> item) {
       int n1 = item.get_id(0);
       int my_igp = item.get_id(1);
       int igp = gpu_indinv[gpu_inv_igp_index[my_igp]];
       sch[n1 * ncouls + my_igp] =
           ComplexType_conj(gpu_aqsmtemp[n1 * ncouls + igp]) *
           gpu_aqsntemp[n1 * ncouls + igp] * 0.5 * gpu_vcoul[igp] *
           gpu_wtilde_array[my_igp * ncouls + igp];
     });
   }).wait();
  end = system_clock::now();
  elapsed = end - start;
  elapsedKernelTimer = elapsed.count();
  std::cout << "********** Pre-process Time Taken **********= "
            << elapsed.count() << " secs" << std::endl;

  auto ach_re0 = malloc_shared<DataType>(1, q);
  auto ach_re1 = malloc_shared<DataType>(1, q);
  auto ach_re2 = malloc_shared<DataType>(1, q);
  auto ach_im0 = malloc_shared<DataType>(1, q);
  auto ach_im1 = malloc_shared<DataType>(1, q);
  auto ach_im2 = malloc_shared<DataType>(1, q);
  ach_re0[0] = 0.00, ach_re1[0] = 0.00, ach_re2[0] = 0.00, ach_im0[0] = 0.00,
  ach_im1[0] = 0.00, ach_im2[0] = 0.00;
  start = system_clock::now();
  int _number_bands =
      number_bands % 64 == 0 ? number_bands : (((number_bands << 6) + 1) >> 6);
  int _ngpown = ngpown % 2 == 0 ? ngpown : ngpown + 1;
  q.submit([&](handler &h) {
     h.parallel_for(
         nd_range<2>(range<2>(_number_bands, _ngpown), range<2>(64, 2)),
         reduction(ach_re0, sycl::plus<>()), reduction(ach_re1, sycl::plus<>()),
         reduction(ach_re2, sycl::plus<>()), reduction(ach_im0, sycl::plus<>()),
         reduction(ach_im1, sycl::plus<>()), reduction(ach_im2, sycl::plus<>()),
         [=](nd_item<2> item, auto &ach_re0, auto &ach_re1, auto &ach_re2,
             auto &ach_im0, auto &ach_im1,
             auto &ach_im2) [[intel::reqd_sub_group_size(32)]] {
           int n1 = item.get_global_id(0);
           int my_igp = item.get_global_id(1);

           if (n1 > number_bands || my_igp > ngpown) return;

           ComplexType sch_store1 = sch[n1 * ncouls + my_igp];

           for (int ig = 0; ig < ncouls; ++ig) {
             auto wdiff =
                 gpu_wx_array[0] - gpu_wtilde_array[my_igp * ncouls + ig];
             auto delw = ComplexType_conj(wdiff) *
                         (1 / (wdiff * ComplexType_conj(wdiff)).real());
             auto sch_array =
                 delw * gpu_wtilde_array[my_igp * ncouls + ig] * sch_store1;
             ach_re0 += (sch_array).real();
             ach_im0 += (sch_array).imag();

             wdiff = gpu_wx_array[1] - gpu_wtilde_array[my_igp * ncouls + ig];
             delw = ComplexType_conj(wdiff) *
                    (1 / (wdiff * ComplexType_conj(wdiff)).real());
             sch_array =
                 delw * gpu_wtilde_array[my_igp * ncouls + ig] * sch_store1;
             ach_re1 += (sch_array).real();
             ach_im1 += (sch_array).imag();

             wdiff = gpu_wx_array[2] - gpu_wtilde_array[my_igp * ncouls + ig];
             delw = ComplexType_conj(wdiff) *
                    (1 / (wdiff * ComplexType_conj(wdiff)).real());
             sch_array =
                 delw * gpu_wtilde_array[my_igp * ncouls + ig] * sch_store1;
             ach_re2 += (sch_array).real();
             ach_im2 += (sch_array).imag();
           }
         });
   }).wait();
  end = system_clock::now();
  elapsed = end - start;
  elapsedKernelTimer = elapsed.count();
  cout << "********** Calculation Time Taken **********= " << elapsed.count()
       << " secs" << std::endl;

  achtemp(0) = ComplexType(ach_re0[0], ach_im0[0]);
  achtemp(1) = ComplexType(ach_re1[0], ach_im1[0]);
  achtemp(2) = ComplexType(ach_re2[0], ach_im2[0]);

  sycl::free(gpu_inv_igp_index, q);
  sycl::free(gpu_indinv, q);
  sycl::free(gpu_vcoul, q);
  sycl::free(gpu_wx_array, q);
  sycl::free(gpu_wtilde_array, q);
  sycl::free(gpu_I_eps_array, q);
  sycl::free(gpu_aqsntemp, q);
  sycl::free(gpu_aqsmtemp, q);
  sycl::free(sch, q);
  sycl::free(ach_im0, q);
  sycl::free(ach_im1, q);
  sycl::free(ach_im2, q);
  sycl::free(ach_re0, q);
  sycl::free(ach_re1, q);
  sycl::free(ach_re2, q);
}
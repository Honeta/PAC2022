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

void noflagOCC_solver(size_t number_bands, size_t ngpown, size_t ncouls,
                      ARRAY1D_int &inv_igp_index, ARRAY1D_int &indinv,
                      ARRAY1D_DataType &wx_array, ARRAY2D &wtilde_array,
                      ARRAY2D &aqsmtemp, ARRAY2D &aqsntemp,
                      ARRAY2D &I_eps_array, ARRAY1D_DataType &vcoul,
                      ARRAY1D &achtemp) {
  auto root_devices = platform(gpu_selector{}).get_devices();

  auto q_0 = queue(
      root_devices[0]
          .create_sub_devices<
              cl::sycl::info::partition_property::partition_by_affinity_domain>(
              cl::sycl::info::partition_affinity_domain::next_partitionable)
              [0]);

  auto q_1 = queue(
      root_devices[0]
          .create_sub_devices<
              cl::sycl::info::partition_property::partition_by_affinity_domain>(
              cl::sycl::info::partition_affinity_domain::next_partitionable)
              [1]);

  auto q_2 = queue(
      root_devices[1]
          .create_sub_devices<
              cl::sycl::info::partition_property::partition_by_affinity_domain>(
              cl::sycl::info::partition_affinity_domain::next_partitionable)
              [0]);

  auto q_3 = queue(
      root_devices[1]
          .create_sub_devices<
              cl::sycl::info::partition_property::partition_by_affinity_domain>(
              cl::sycl::info::partition_affinity_domain::next_partitionable)
              [1]);

  int _number_bands =
      number_bands % 256 == 0 ? number_bands : (((number_bands << 8) + 1) >> 8);
  int _ngpown = ngpown % 2 == 0 ? ngpown : ngpown + 1;
  int number_bands_per_tile = _number_bands >> 2;

  DataType *ach_re0_0 = malloc_shared<DataType>(1, q_0);
  DataType *ach_re1_0 = malloc_shared<DataType>(1, q_0);
  DataType *ach_re2_0 = malloc_shared<DataType>(1, q_0);
  DataType *ach_im0_0 = malloc_shared<DataType>(1, q_0);
  DataType *ach_im1_0 = malloc_shared<DataType>(1, q_0);
  DataType *ach_im2_0 = malloc_shared<DataType>(1, q_0);
  ach_re0_0[0] = 0.00, ach_re1_0[0] = 0.00, ach_re2_0[0] = 0.00,
  ach_im0_0[0] = 0.00, ach_im1_0[0] = 0.00, ach_im2_0[0] = 0.00;

  DataType *ach_re0_1 = malloc_shared<DataType>(1, q_1);
  DataType *ach_re1_1 = malloc_shared<DataType>(1, q_1);
  DataType *ach_re2_1 = malloc_shared<DataType>(1, q_1);
  DataType *ach_im0_1 = malloc_shared<DataType>(1, q_1);
  DataType *ach_im1_1 = malloc_shared<DataType>(1, q_1);
  DataType *ach_im2_1 = malloc_shared<DataType>(1, q_1);
  ach_re0_1[0] = 0.00, ach_re1_1[0] = 0.00, ach_re2_1[0] = 0.00,
  ach_im0_1[0] = 0.00, ach_im1_1[0] = 0.00, ach_im2_1[0] = 0.00;

  DataType *ach_re0_2 = malloc_shared<DataType>(1, q_2);
  DataType *ach_re1_2 = malloc_shared<DataType>(1, q_2);
  DataType *ach_re2_2 = malloc_shared<DataType>(1, q_2);
  DataType *ach_im0_2 = malloc_shared<DataType>(1, q_2);
  DataType *ach_im1_2 = malloc_shared<DataType>(1, q_2);
  DataType *ach_im2_2 = malloc_shared<DataType>(1, q_2);
  ach_re0_2[0] = 0.00, ach_re1_2[0] = 0.00, ach_re2_2[0] = 0.00,
  ach_im0_2[0] = 0.00, ach_im1_2[0] = 0.00, ach_im2_2[0] = 0.00;

  DataType *ach_re0_3 = malloc_shared<DataType>(1, q_3);
  DataType *ach_re1_3 = malloc_shared<DataType>(1, q_3);
  DataType *ach_re2_3 = malloc_shared<DataType>(1, q_3);
  DataType *ach_im0_3 = malloc_shared<DataType>(1, q_3);
  DataType *ach_im1_3 = malloc_shared<DataType>(1, q_3);
  DataType *ach_im2_3 = malloc_shared<DataType>(1, q_3);
  ach_re0_3[0] = 0.00, ach_re1_3[0] = 0.00, ach_re2_3[0] = 0.00,
  ach_im0_3[0] = 0.00, ach_im1_3[0] = 0.00, ach_im2_3[0] = 0.00;

  // Memory allocation

  int *gpu_inv_igp_index_0 = malloc_device<int>(ngpown, q_0);
  int *gpu_indinv_0 = malloc_device<int>(ngpown, q_0);
  DataType *gpu_vcoul_0 = malloc_device<DataType>(ncouls, q_0);
  DataType *gpu_wx_array_0 = malloc_device<DataType>(nend - nstart, q_0);
  ComplexType *gpu_wtilde_array_0 =
      malloc_device<ComplexType>(ngpown * ncouls, q_0);
  ComplexType *gpu_I_eps_array_0 =
      malloc_device<ComplexType>(ngpown * ncouls, q_0);
  ComplexType *gpu_aqsntemp_0 =
      malloc_device<ComplexType>(number_bands_per_tile * ncouls, q_0);
  ComplexType *gpu_aqsmtemp_0 =
      malloc_device<ComplexType>(number_bands_per_tile * ncouls, q_0);
  ComplexType *sch_0 =
      malloc_device<ComplexType>(number_bands_per_tile * ncouls, q_0);

  int *gpu_inv_igp_index_1 = malloc_device<int>(ngpown, q_1);
  int *gpu_indinv_1 = malloc_device<int>(ngpown, q_1);
  DataType *gpu_vcoul_1 = malloc_device<DataType>(ncouls, q_1);
  DataType *gpu_wx_array_1 = malloc_device<DataType>(nend - nstart, q_1);
  ComplexType *gpu_wtilde_array_1 =
      malloc_device<ComplexType>(ngpown * ncouls, q_1);
  ComplexType *gpu_I_eps_array_1 =
      malloc_device<ComplexType>(ngpown * ncouls, q_1);
  ComplexType *gpu_aqsntemp_1 =
      malloc_device<ComplexType>(number_bands_per_tile * ncouls, q_1);
  ComplexType *gpu_aqsmtemp_1 =
      malloc_device<ComplexType>(number_bands_per_tile * ncouls, q_1);
  ComplexType *sch_1 =
      malloc_device<ComplexType>(number_bands_per_tile * ncouls, q_1);

  int *gpu_inv_igp_index_2 = malloc_device<int>(ngpown, q_2);
  int *gpu_indinv_2 = malloc_device<int>(ngpown, q_2);
  DataType *gpu_vcoul_2 = malloc_device<DataType>(ncouls, q_2);
  DataType *gpu_wx_array_2 = malloc_device<DataType>(nend - nstart, q_2);
  ComplexType *gpu_wtilde_array_2 =
      malloc_device<ComplexType>(ngpown * ncouls, q_2);
  ComplexType *gpu_I_eps_array_2 =
      malloc_device<ComplexType>(ngpown * ncouls, q_2);
  ComplexType *gpu_aqsntemp_2 =
      malloc_device<ComplexType>(number_bands_per_tile * ncouls, q_2);
  ComplexType *gpu_aqsmtemp_2 =
      malloc_device<ComplexType>(number_bands_per_tile * ncouls, q_2);
  ComplexType *sch_2 =
      malloc_device<ComplexType>(number_bands_per_tile * ncouls, q_2);

  int *gpu_inv_igp_index_3 = malloc_device<int>(ngpown, q_3);
  int *gpu_indinv_3 = malloc_device<int>(ngpown, q_3);
  DataType *gpu_vcoul_3 = malloc_device<DataType>(ncouls, q_3);
  DataType *gpu_wx_array_3 = malloc_device<DataType>(nend - nstart, q_3);
  ComplexType *gpu_wtilde_array_3 =
      malloc_device<ComplexType>(ngpown * ncouls, q_3);
  ComplexType *gpu_I_eps_array_3 =
      malloc_device<ComplexType>(ngpown * ncouls, q_3);
  ComplexType *gpu_aqsntemp_3 =
      malloc_device<ComplexType>(number_bands_per_tile * ncouls, q_3);
  ComplexType *gpu_aqsmtemp_3 =
      malloc_device<ComplexType>(number_bands_per_tile * ncouls, q_3);
  ComplexType *sch_3 =
      malloc_device<ComplexType>(number_bands_per_tile * ncouls, q_3);

  // Copy & Pre-process

  q_0.memcpy(gpu_inv_igp_index_0, inv_igp_index.dptr, ngpown * sizeof(int))
      .wait();
  q_0.memcpy(gpu_indinv_0, indinv.dptr, ngpown * sizeof(int)).wait();
  q_0.memcpy(gpu_vcoul_0, vcoul.dptr, ngpown * sizeof(DataType)).wait();
  q_0.memcpy(gpu_wtilde_array_0, wtilde_array.dptr,
             ngpown * ncouls * sizeof(ComplexType))
      .wait();
  q_0.memcpy(gpu_aqsntemp_0, aqsntemp.dptr,
             number_bands_per_tile * ncouls * sizeof(ComplexType))
      .wait();
  q_0.memcpy(gpu_aqsmtemp_0, aqsmtemp.dptr,
             number_bands_per_tile * ncouls * sizeof(ComplexType))
      .wait();

  auto e_0_1 = q_0.submit([&](handler &h) {
    h.parallel_for(range<2>(number_bands_per_tile, ngpown), [=](item<2> item) {
      int n1 = item.get_id(0);
      int my_igp = item.get_id(1);
      int igp = gpu_indinv_0[gpu_inv_igp_index_0[my_igp]];
      sch_0[n1 * ncouls + my_igp] =
          ComplexType_conj(gpu_aqsmtemp_0[n1 * ncouls + igp]) *
          gpu_aqsntemp_0[n1 * ncouls + igp] * 0.5 * gpu_vcoul_0[igp] *
          gpu_wtilde_array_0[my_igp * ncouls + igp];
    });
  });

  q_0.memcpy(gpu_wx_array_0, wx_array.dptr, (nend - nstart) * sizeof(DataType))
      .wait();
  q_0.memcpy(gpu_I_eps_array_0, I_eps_array.dptr,
             ngpown * ncouls * sizeof(ComplexType))
      .wait();

  q_1.memcpy(gpu_inv_igp_index_1, inv_igp_index.dptr, ngpown * sizeof(int))
      .wait();
  q_1.memcpy(gpu_indinv_1, indinv.dptr, ngpown * sizeof(int)).wait();
  q_1.memcpy(gpu_vcoul_1, vcoul.dptr, ngpown * sizeof(DataType)).wait();
  q_1.memcpy(gpu_wtilde_array_1, wtilde_array.dptr,
             ngpown * ncouls * sizeof(ComplexType))
      .wait();
  q_1.memcpy(gpu_aqsntemp_1, aqsntemp.dptr + number_bands_per_tile,
             number_bands_per_tile * ncouls * sizeof(ComplexType))
      .wait();
  q_1.memcpy(gpu_aqsmtemp_1, aqsmtemp.dptr + number_bands_per_tile,
             number_bands_per_tile * ncouls * sizeof(ComplexType))
      .wait();

  auto e_1_1 = q_1.submit([&](handler &h) {
    h.parallel_for(range<2>(number_bands_per_tile, ngpown), [=](item<2> item) {
      int n1 = item.get_id(0);
      int my_igp = item.get_id(1);
      int igp = gpu_indinv_1[gpu_inv_igp_index_1[my_igp]];
      sch_1[n1 * ncouls + my_igp] =
          ComplexType_conj(gpu_aqsmtemp_1[n1 * ncouls + igp]) *
          gpu_aqsntemp_1[n1 * ncouls + igp] * 0.5 * gpu_vcoul_1[igp] *
          gpu_wtilde_array_1[my_igp * ncouls + igp];
    });
  });

  q_1.memcpy(gpu_wx_array_1, wx_array.dptr, (nend - nstart) * sizeof(DataType))
      .wait();
  q_1.memcpy(gpu_I_eps_array_1, I_eps_array.dptr,
             ngpown * ncouls * sizeof(ComplexType))
      .wait();

  q_2.memcpy(gpu_inv_igp_index_2, inv_igp_index.dptr, ngpown * sizeof(int))
      .wait();
  q_2.memcpy(gpu_indinv_2, indinv.dptr, ngpown * sizeof(int)).wait();
  q_2.memcpy(gpu_vcoul_2, vcoul.dptr, ngpown * sizeof(DataType)).wait();
  q_2.memcpy(gpu_wtilde_array_2, wtilde_array.dptr,
             ngpown * ncouls * sizeof(ComplexType))
      .wait();
  q_2.memcpy(gpu_aqsntemp_2, aqsntemp.dptr + number_bands_per_tile * 2,
             number_bands_per_tile * ncouls * sizeof(ComplexType))
      .wait();
  q_2.memcpy(gpu_aqsmtemp_2, aqsmtemp.dptr + number_bands_per_tile * 2,
             number_bands_per_tile * ncouls * sizeof(ComplexType))
      .wait();

  auto e_2_1 = q_2.submit([&](handler &h) {
    h.parallel_for(range<2>(number_bands_per_tile, ngpown), [=](item<2> item) {
      int n1 = item.get_id(0);
      int my_igp = item.get_id(1);
      int igp = gpu_indinv_2[gpu_inv_igp_index_2[my_igp]];
      sch_2[n1 * ncouls + my_igp] =
          ComplexType_conj(gpu_aqsmtemp_2[n1 * ncouls + igp]) *
          gpu_aqsntemp_2[n1 * ncouls + igp] * 0.5 * gpu_vcoul_2[igp] *
          gpu_wtilde_array_2[my_igp * ncouls + igp];
    });
  });

  q_2.memcpy(gpu_wx_array_2, wx_array.dptr, (nend - nstart) * sizeof(DataType))
      .wait();
  q_2.memcpy(gpu_I_eps_array_2, I_eps_array.dptr,
             ngpown * ncouls * sizeof(ComplexType))
      .wait();

  q_3.memcpy(gpu_inv_igp_index_3, inv_igp_index.dptr, ngpown * sizeof(int))
      .wait();
  q_3.memcpy(gpu_indinv_3, indinv.dptr, ngpown * sizeof(int)).wait();
  q_3.memcpy(gpu_vcoul_3, vcoul.dptr, ngpown * sizeof(DataType)).wait();
  q_3.memcpy(gpu_wtilde_array_3, wtilde_array.dptr,
             ngpown * ncouls * sizeof(ComplexType))
      .wait();
  q_3.memcpy(gpu_aqsntemp_3, aqsntemp.dptr + number_bands_per_tile * 3,
             number_bands_per_tile * ncouls * sizeof(ComplexType))
      .wait();
  q_3.memcpy(gpu_aqsmtemp_3, aqsmtemp.dptr + number_bands_per_tile * 3,
             number_bands_per_tile * ncouls * sizeof(ComplexType))
      .wait();

  auto e_3_1 = q_3.submit([&](handler &h) {
    h.parallel_for(range<2>(number_bands_per_tile, ngpown), [=](item<2> item) {
      int n1 = item.get_id(0);
      int my_igp = item.get_id(1);
      int igp = gpu_indinv_3[gpu_inv_igp_index_3[my_igp]];
      sch_3[n1 * ncouls + my_igp] =
          ComplexType_conj(gpu_aqsmtemp_3[n1 * ncouls + igp]) *
          gpu_aqsntemp_3[n1 * ncouls + igp] * 0.5 * gpu_vcoul_3[igp] *
          gpu_wtilde_array_3[my_igp * ncouls + igp];
    });
  });

  q_3.memcpy(gpu_wx_array_3, wx_array.dptr, (nend - nstart) * sizeof(DataType))
      .wait();
  q_3.memcpy(gpu_I_eps_array_3, I_eps_array.dptr,
             ngpown * ncouls * sizeof(ComplexType))
      .wait();

  // Calculation

  auto e_0_2 = q_0.submit([&](handler &h) {
    h.depends_on(e_0_1);
    h.parallel_for(
        nd_range<2>(range<2>(number_bands_per_tile, _ngpown), range<2>(64, 2)),
        reduction(ach_re0_0, sycl::plus<>()),
        reduction(ach_re1_0, sycl::plus<>()),
        reduction(ach_re2_0, sycl::plus<>()),
        reduction(ach_im0_0, sycl::plus<>()),
        reduction(ach_im1_0, sycl::plus<>()),
        reduction(ach_im2_0, sycl::plus<>()),
        [=](nd_item<2> item, auto &ach_re0_0, auto &ach_re1_0, auto &ach_re2_0,
            auto &ach_im0_0, auto &ach_im1_0,
            auto &ach_im2_0) [[intel::reqd_sub_group_size(32)]] {
          int my_igp = item.get_global_id(1);

          if (item.get_global_id(0) >= number_bands || my_igp >= ngpown) return;

          ComplexType sch_store1 =
              sch_0[item.get_global_id(0) * ncouls + my_igp];

          for (int ig = 0; ig < ncouls; ++ig) {
            auto wdiff =
                gpu_wx_array_0[0] - gpu_wtilde_array_0[my_igp * ncouls + ig];
            auto delw = ComplexType_conj(wdiff) *
                        (1 / (wdiff * ComplexType_conj(wdiff)).real());
            auto sch_array =
                delw * gpu_wtilde_array_0[my_igp * ncouls + ig] * sch_store1;
            ach_re0_0 += (sch_array).real();
            ach_im0_0 += (sch_array).imag();

            wdiff =
                gpu_wx_array_0[1] - gpu_wtilde_array_0[my_igp * ncouls + ig];
            delw = ComplexType_conj(wdiff) *
                   (1 / (wdiff * ComplexType_conj(wdiff)).real());
            sch_array =
                delw * gpu_wtilde_array_0[my_igp * ncouls + ig] * sch_store1;
            ach_re1_0 += (sch_array).real();
            ach_im1_0 += (sch_array).imag();

            wdiff =
                gpu_wx_array_0[2] - gpu_wtilde_array_0[my_igp * ncouls + ig];
            delw = ComplexType_conj(wdiff) *
                   (1 / (wdiff * ComplexType_conj(wdiff)).real());
            sch_array =
                delw * gpu_wtilde_array_0[my_igp * ncouls + ig] * sch_store1;
            ach_re2_0 += (sch_array).real();
            ach_im2_0 += (sch_array).imag();
          }
        });
  });

  auto e_1_2 = q_1.submit([&](handler &h) {
    h.depends_on(e_1_1);
    h.parallel_for(
        nd_range<2>(range<2>(number_bands_per_tile, _ngpown), range<2>(64, 2)),
        reduction(ach_re0_1, sycl::plus<>()),
        reduction(ach_re1_1, sycl::plus<>()),
        reduction(ach_re2_1, sycl::plus<>()),
        reduction(ach_im0_1, sycl::plus<>()),
        reduction(ach_im1_1, sycl::plus<>()),
        reduction(ach_im2_1, sycl::plus<>()),
        [=](nd_item<2> item, auto &ach_re0_1, auto &ach_re1_1, auto &ach_re2_1,
            auto &ach_im0_1, auto &ach_im1_1,
            auto &ach_im2_1) [[intel::reqd_sub_group_size(32)]] {
          int my_igp = item.get_global_id(1);

          if (item.get_global_id(0) + number_bands_per_tile >= number_bands ||
              my_igp >= ngpown)
            return;

          ComplexType sch_store1 =
              sch_1[item.get_global_id(0) * ncouls + my_igp];

          for (int ig = 0; ig < ncouls; ++ig) {
            auto wdiff =
                gpu_wx_array_1[0] - gpu_wtilde_array_1[my_igp * ncouls + ig];
            auto delw = ComplexType_conj(wdiff) *
                        (1 / (wdiff * ComplexType_conj(wdiff)).real());
            auto sch_array =
                delw * gpu_wtilde_array_1[my_igp * ncouls + ig] * sch_store1;
            ach_re0_1 += (sch_array).real();
            ach_im0_1 += (sch_array).imag();

            wdiff =
                gpu_wx_array_1[1] - gpu_wtilde_array_1[my_igp * ncouls + ig];
            delw = ComplexType_conj(wdiff) *
                   (1 / (wdiff * ComplexType_conj(wdiff)).real());
            sch_array =
                delw * gpu_wtilde_array_1[my_igp * ncouls + ig] * sch_store1;
            ach_re1_1 += (sch_array).real();
            ach_im1_1 += (sch_array).imag();

            wdiff =
                gpu_wx_array_1[2] - gpu_wtilde_array_1[my_igp * ncouls + ig];
            delw = ComplexType_conj(wdiff) *
                   (1 / (wdiff * ComplexType_conj(wdiff)).real());
            sch_array =
                delw * gpu_wtilde_array_1[my_igp * ncouls + ig] * sch_store1;
            ach_re2_1 += (sch_array).real();
            ach_im2_1 += (sch_array).imag();
          }
        });
  });

  auto e_2_2 = q_2.submit([&](handler &h) {
    h.depends_on(e_2_1);
    h.parallel_for(
        nd_range<2>(range<2>(number_bands_per_tile, _ngpown), range<2>(64, 2)),
        reduction(ach_re0_2, sycl::plus<>()),
        reduction(ach_re1_2, sycl::plus<>()),
        reduction(ach_re2_2, sycl::plus<>()),
        reduction(ach_im0_2, sycl::plus<>()),
        reduction(ach_im1_2, sycl::plus<>()),
        reduction(ach_im2_2, sycl::plus<>()),
        [=](nd_item<2> item, auto &ach_re0_2, auto &ach_re1_2, auto &ach_re2_2,
            auto &ach_im0_2, auto &ach_im1_2,
            auto &ach_im2_2) [[intel::reqd_sub_group_size(32)]] {
          int my_igp = item.get_global_id(1);

          if (item.get_global_id(0) + number_bands_per_tile * 2 >=
                  number_bands ||
              my_igp >= ngpown)
            return;

          ComplexType sch_store1 =
              sch_2[item.get_global_id(0) * ncouls + my_igp];

          for (int ig = 0; ig < ncouls; ++ig) {
            auto wdiff =
                gpu_wx_array_2[0] - gpu_wtilde_array_2[my_igp * ncouls + ig];
            auto delw = ComplexType_conj(wdiff) *
                        (1 / (wdiff * ComplexType_conj(wdiff)).real());
            auto sch_array =
                delw * gpu_wtilde_array_2[my_igp * ncouls + ig] * sch_store1;
            ach_re0_2 += (sch_array).real();
            ach_im0_2 += (sch_array).imag();

            wdiff =
                gpu_wx_array_2[1] - gpu_wtilde_array_2[my_igp * ncouls + ig];
            delw = ComplexType_conj(wdiff) *
                   (1 / (wdiff * ComplexType_conj(wdiff)).real());
            sch_array =
                delw * gpu_wtilde_array_2[my_igp * ncouls + ig] * sch_store1;
            ach_re1_2 += (sch_array).real();
            ach_im1_2 += (sch_array).imag();

            wdiff =
                gpu_wx_array_2[2] - gpu_wtilde_array_2[my_igp * ncouls + ig];
            delw = ComplexType_conj(wdiff) *
                   (1 / (wdiff * ComplexType_conj(wdiff)).real());
            sch_array =
                delw * gpu_wtilde_array_2[my_igp * ncouls + ig] * sch_store1;
            ach_re2_2 += (sch_array).real();
            ach_im2_2 += (sch_array).imag();
          }
        });
  });

  auto e_3_2 = q_3.submit([&](handler &h) {
    h.depends_on(e_3_1);
    h.parallel_for(
        nd_range<2>(range<2>(number_bands_per_tile, _ngpown), range<2>(64, 2)),
        reduction(ach_re0_3, sycl::plus<>()),
        reduction(ach_re1_3, sycl::plus<>()),
        reduction(ach_re2_3, sycl::plus<>()),
        reduction(ach_im0_3, sycl::plus<>()),
        reduction(ach_im1_3, sycl::plus<>()),
        reduction(ach_im2_3, sycl::plus<>()),
        [=](nd_item<2> item, auto &ach_re0_3, auto &ach_re1_3, auto &ach_re2_3,
            auto &ach_im0_3, auto &ach_im1_3,
            auto &ach_im2_3) [[intel::reqd_sub_group_size(32)]] {
          int my_igp = item.get_global_id(1);

          if (item.get_global_id(0) + number_bands_per_tile * 3 >=
                  number_bands ||
              my_igp >= ngpown)
            return;

          ComplexType sch_store1 =
              sch_3[item.get_global_id(0) * ncouls + my_igp];

          for (int ig = 0; ig < ncouls; ++ig) {
            auto wdiff =
                gpu_wx_array_3[0] - gpu_wtilde_array_3[my_igp * ncouls + ig];
            auto delw = ComplexType_conj(wdiff) *
                        (1 / (wdiff * ComplexType_conj(wdiff)).real());
            auto sch_array =
                delw * gpu_wtilde_array_3[my_igp * ncouls + ig] * sch_store1;
            ach_re0_3 += (sch_array).real();
            ach_im0_3 += (sch_array).imag();

            wdiff =
                gpu_wx_array_3[1] - gpu_wtilde_array_3[my_igp * ncouls + ig];
            delw = ComplexType_conj(wdiff) *
                   (1 / (wdiff * ComplexType_conj(wdiff)).real());
            sch_array =
                delw * gpu_wtilde_array_3[my_igp * ncouls + ig] * sch_store1;
            ach_re1_3 += (sch_array).real();
            ach_im1_3 += (sch_array).imag();

            wdiff =
                gpu_wx_array_3[2] - gpu_wtilde_array_3[my_igp * ncouls + ig];
            delw = ComplexType_conj(wdiff) *
                   (1 / (wdiff * ComplexType_conj(wdiff)).real());
            sch_array =
                delw * gpu_wtilde_array_3[my_igp * ncouls + ig] * sch_store1;
            ach_re2_3 += (sch_array).real();
            ach_im2_3 += (sch_array).imag();
          }
        });
  });

  e_0_2.wait();
  e_1_2.wait();
  e_2_2.wait();
  e_3_2.wait();

  achtemp(0) =
      ComplexType(ach_re0_0[0] + ach_re0_1[0] + ach_re0_2[0] + ach_re0_3[0],
                  ach_im0_0[0] + ach_im0_1[0] + ach_im0_2[0] + ach_im0_3[0]);
  achtemp(1) =
      ComplexType(ach_re1_0[0] + ach_re1_1[0] + ach_re1_2[0] + ach_re1_3[0],
                  ach_im1_0[0] + ach_im1_1[0] + ach_im1_2[0] + ach_im1_3[0]);
  achtemp(2) =
      ComplexType(ach_re2_0[0] + ach_re2_1[0] + ach_re2_2[0] + ach_re2_3[0],
                  ach_im2_0[0] + ach_im2_1[0] + ach_im2_2[0] + ach_im2_3[0]);

  sycl::free(gpu_inv_igp_index_0, q_0);
  sycl::free(gpu_indinv_0, q_0);
  sycl::free(gpu_vcoul_0, q_0);
  sycl::free(gpu_wx_array_0, q_0);
  sycl::free(gpu_wtilde_array_0, q_0);
  sycl::free(gpu_I_eps_array_0, q_0);
  sycl::free(gpu_aqsntemp_0, q_0);
  sycl::free(gpu_aqsmtemp_0, q_0);
  sycl::free(sch_0, q_0);
  sycl::free(ach_im0_0, q_0);
  sycl::free(ach_im1_0, q_0);
  sycl::free(ach_im2_0, q_0);
  sycl::free(ach_re0_0, q_0);
  sycl::free(ach_re1_0, q_0);
  sycl::free(ach_re2_0, q_0);

  sycl::free(gpu_inv_igp_index_1, q_1);
  sycl::free(gpu_indinv_1, q_1);
  sycl::free(gpu_vcoul_1, q_1);
  sycl::free(gpu_wx_array_1, q_1);
  sycl::free(gpu_wtilde_array_1, q_1);
  sycl::free(gpu_I_eps_array_1, q_1);
  sycl::free(gpu_aqsntemp_1, q_1);
  sycl::free(gpu_aqsmtemp_1, q_1);
  sycl::free(sch_1, q_1);
  sycl::free(ach_im0_1, q_1);
  sycl::free(ach_im1_1, q_1);
  sycl::free(ach_im2_1, q_1);
  sycl::free(ach_re0_1, q_1);
  sycl::free(ach_re1_1, q_1);
  sycl::free(ach_re2_1, q_1);

  sycl::free(gpu_inv_igp_index_2, q_2);
  sycl::free(gpu_indinv_2, q_2);
  sycl::free(gpu_vcoul_2, q_2);
  sycl::free(gpu_wx_array_2, q_2);
  sycl::free(gpu_wtilde_array_2, q_2);
  sycl::free(gpu_I_eps_array_2, q_2);
  sycl::free(gpu_aqsntemp_2, q_2);
  sycl::free(gpu_aqsmtemp_2, q_2);
  sycl::free(sch_2, q_2);
  sycl::free(ach_im0_2, q_2);
  sycl::free(ach_im1_2, q_2);
  sycl::free(ach_im2_2, q_2);
  sycl::free(ach_re0_2, q_2);
  sycl::free(ach_re1_2, q_2);
  sycl::free(ach_re2_2, q_2);

  sycl::free(gpu_inv_igp_index_3, q_3);
  sycl::free(gpu_indinv_3, q_3);
  sycl::free(gpu_vcoul_3, q_3);
  sycl::free(gpu_wx_array_3, q_3);
  sycl::free(gpu_wtilde_array_3, q_3);
  sycl::free(gpu_I_eps_array_3, q_3);
  sycl::free(gpu_aqsntemp_3, q_3);
  sycl::free(gpu_aqsmtemp_3, q_3);
  sycl::free(sch_3, q_3);
  sycl::free(ach_im0_3, q_3);
  sycl::free(ach_im1_3, q_3);
  sycl::free(ach_im2_3, q_3);
  sycl::free(ach_re0_3, q_3);
  sycl::free(ach_re1_3, q_3);
  sycl::free(ach_re2_3, q_3);
}
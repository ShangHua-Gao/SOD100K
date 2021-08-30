from libcpp.vector cimport vector
from libcpp.string cimport string
cdef extern from "sal_metric.hpp":
    cdef cppclass SalMetric:
        SalMetric() except +
        SalMetric(int num_thread) except +
        void load_list(vector[string] sal_lst, vector[string] gt_lst)
        void do_evaluation()
        void set_num_thread(int num_thread)

cdef class PySalMetric:
    cdef SalMetric* thisptr
    def __cinit__(self, int num_thread=4):
        self.thisptr = new SalMetric(num_thread)

    def __dealloc__(self):
        del self.thisptr

    def load_list(self, sal_lst, gt_lst):
        self.thisptr.load_list(sal_lst, gt_lst)

    def do_evaluation(self):
        self.thisptr.do_evaluation()

def do_evaluation(num_thread, sal_lst, gt_lst):
    assert isinstance(sal_lst, list)
    assert isinstance(gt_lst, list)
    cdef SalMetric salmetric
    salmetric.set_num_thread(num_thread)
    salmetric.load_list(sal_lst, gt_lst)
    salmetric.do_evaluation()


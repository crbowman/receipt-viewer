import re, os, time, pickle, collections
from nltk.tokenize import word_tokenize

FILE_REGEX = re.compile('\.eml$|\.txt$', re.IGNORECASE)

def with_each_file_name(file_names, handler, offset=0, limit=-1, debug=False, extra_args=None):
    '''
    :param folder_path:
    :param handler: a callback method that takes a file_name as parameter
        and does some processing on that file. If it returns something the
        result is appended to the resulting array
    :param file_regex:
    :param limit: If -1 it'll process all the files
    :return:
    '''
    if (debug):
        print("[%r] Processing files in folder... %d -> %d" % (os.getpid(), offset, offset + limit))
    results = []
    for index, file_name in enumerate(file_names):
        if (debug and index > offset and index % 10 == 0):
            print("[%r] Processed already %d -> %d" % (os.getpid(), offset, index - offset))
        if index >= offset:
            if not extra_args:
                res = handler(file_name)
            else:
                res = handler(file_name, extra_args)
            if res:
                results.append(res)
        if limit != -1 and index + 1 >= offset + limit:
            break
    return results


def hexa_sort(f_name):
    temp = f_name.split('.')[0]
    try:
        return int(temp, 16)
    except :
        return 0


def files_in(folder_path, file_regex=FILE_REGEX):
    return [f for f in sorted(os.listdir(folder_path), key=hexa_sort, reverse=True) if file_regex.search(f)]

def load_tagger(file_name = 'out/c_tagger.pickle'):
    with open(file_name, 'rb') as in_file:
        tagger = pickle.load(in_file)
    return tagger

TAGGER = load_tagger()

def tagged_sents(text, tagger = TAGGER):
    if not tagger:
        tagger = load_tagger()
    return tagger.tag(word_tokenize(text))

from multiprocessing import Process, Lock, Value, Queue


def process_in_batches(method_handle, extra_args=None, batch_size=10, parallel_processes=8, debug=False):
    results_queue = Queue()
    lock = Lock()
    active_process_counter = Value('i', parallel_processes)
    if debug:
        print("[%r] -- Parallel processing with: %d processes" % (os.getpid(), active_process_counter.value))
    for index in range(parallel_processes):
        context = BatchContext(lock, active_process_counter, results_queue, index * batch_size, batch_size, extra_args)
        p = Process(target=method_handle, args=(context,))
        p.start()
    while active_process_counter.value > 0:
        time.sleep(1)
    results = []
    for i in range(parallel_processes):
        results.extend(results_queue.get(True))
    return results


class BatchContext():
    def __init__(self, lock, active_process_counter, results_queue, offset, batch_size, extra_args, debug=False):
        self.lock = lock
        self.active_process_counter = active_process_counter
        self.results_queue = results_queue
        self.offset = offset
        self.batch_size = batch_size
        self.extra_args = extra_args
        self.debug = debug

    def finished_batch_with_results(self, results):
        self.lock.acquire()
        self.results_queue.put(results)
        self.active_process_counter.value -= 1
        self.lock.release()
        if self.debug:
            print('[%r] Finished with status: %r' % (os.getpid(), self.__str__()))

    def __str__(self):
        return 'active: {}, offset: {}, batch size: {}' \
            .format(self.active_process_counter.value, self.offset, self.batch_size)

class Debugger():
    def __init__(self, debug_enabled = False):
        self.debug_enabled = debug_enabled

    def debug(self, obj):
        if self.debug_enabled:
            if isinstance(obj, collections.Iterable) and not isinstance(obj, str):
                for o in obj:
                    print(o)
            else:
                print(obj)
        print()
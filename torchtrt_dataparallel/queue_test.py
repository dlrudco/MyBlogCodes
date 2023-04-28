from multiprocessing import Process, Queue

def test_queue():
    inpq = Queue()
    outq = Queue()
    p1 = Process(target=pusher, args=(inpq,outq,))
    p2 = Process(target=popper, args=(inpq,outq,))
    p1.start()
    p2.start()
    
def pusher(inpq, outq):
    for i in range(10):
        inpq.put(i)
        print(outq.get())
    inpq.put(None)
    
def popper(inpq, outq):
    while True:
        item = inpq.get()
        if item is None:
            break
        else:
            outq.put(item)

if __name__ == "__main__":
    test_queue()
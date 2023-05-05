from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
import os


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
        # except:
        #     print("Issue processing data with path: ", data['file'])
        #     if (opt.cleanup_mode == 1):
        #         os.remove(r"{path2}".format(path2=data['file'])[2:-2])
        
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()

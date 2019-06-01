import os
import matplotlib.pylab as plt


def calc_number_of_files_per_category(path):
    dirs = [dict(path=os.path.join(path, d)) for d in os.listdir(path) if
            (not d.startswith('.')) and os.path.isdir(os.path.join(path, d))]
    # min_number_of_files = 10000
    for dir_obj in dirs:
        amount_of_files = len([name for name in os.listdir(dir_obj["path"]) if
                               (not name.startswith('.')) and os.path.isfile(os.path.join(dir_obj["path"], name))])
        dir_obj["number"] = amount_of_files

        # if amount_of_files < min_number_of_files:
        #     min_number_of_files = amount_of_files

    return dirs

def present_plot(details):
    lists = sorted(details.items())
    x, y = zip(*lists)
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    # Change Path for you
    PATH = "../data/train"
    dir_params = (calc_number_of_files_per_category(PATH))
    details = dict()
    list_details = []
    for elem in dir_params:
        if elem['number'] in details:
            details[elem['number']] += 1
        else: details[elem['number']] = 1
        # details[elem['number']] = details.get(elem['number'] + 1,1)
        list_details.append(elem['number'])

    list_details.sort()
    print(list_details)
    print(details)

    present_plot(details)

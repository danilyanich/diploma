import argparse
import sys
import pickle
import utils as ut


def get_file_data(in_file):
  data = pickle.load(in_file)

  method_name = data.get('method_name')
  decomposition_rank = data.get('decomposition_rank')
  errors_trace = data.get('errors_trace')
  file_name = ut.get_filename(in_file.name)

  column_name = '{}.{}.{}'.format(file_name, decomposition_rank, method_name)

  return column_name, errors_trace


def save_csv_table(columns, errors, file):
  columns_len = len(columns)
  max_row_length = max([len(v) for v in errors])

  table_columns = ['iteration'] + columns
  header = ','.join(['{:>30}'.format(h) for h in table_columns])

  out_file.write(header)
  out_file.write('\n')

  for iteration in range(max_row_length):
    row_values = [column[iteration] if len(column) > iteration else None for column in errors]
    row_data = [iteration] + ['{:.5f}'.format(v) if v is not None else '' for v in row_values]
    row = ','.join(['{:>30}'.format(h) for h in row_data])

    out_file.write(row)
    out_file.write('\n')
  pass


parser = argparse.ArgumentParser()

parser.add_argument('input', nargs='+')
parser.add_argument('--out', default=None)


if __name__ == '__main__':
  args = parser.parse_args()

  in_files_paths = args.input
  out_file_path = args.out

  in_files = [open(path, mode='rb+') for path in in_files_paths]
  out_file = open(out_file_path, mode='w+') if out_file_path else sys.stdout

  raw_data = [get_file_data(in_file) for in_file in in_files]
  columns_tuple, values_tuple = zip(*raw_data)
  columns, rows = list(columns_tuple), list(values_tuple)


  save_csv_table(columns, rows, out_file)


  pass

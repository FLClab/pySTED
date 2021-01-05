# this little script will only be used to add events to files, so I don't have to do this
# in the test script file, so I don't write the same event 1000 times in one file

# juste faire une fonction dans utils pour ça?
from pysted import utils

path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1_events.txt"

frame_start, frame_end = 450, 490
col_start, col_end = 140, 160
row_start, row_end = 100, 125

event = {"frame start": frame_start,
         "frame end": frame_end,
         "col_start": col_start,
         "col_end": col_end,
         "row_start": row_start,
         "row_end": row_end}

list_pre = utils.event_reader(path)
print(f"list_pre : {list_pre}")

utils.dict_write_func(path, event)

list_post = utils.event_reader(path)
print(f"lits_post = {list_post}")
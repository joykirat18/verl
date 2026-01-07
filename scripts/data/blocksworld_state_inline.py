# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os
import uuid


import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="blocksworld_state_inline")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--is_eval", default=False, action="store_true")


    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "divelab/blocksworld"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop('question')
            answer = example.pop('answer')
            solution = example.pop('solution')

            prompt = """I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do

Pick up a block
Unstack a block from on top of another block
Put down a block
Stack a block on top of another block

I have the following restrictions on my actions:
I can only pick up or unstack one block at a time.
I can only pick up or unstack a block if my hand is empty.
I can only pick up a block if the block is on the table and the block is clear.
A block is clear if the block has no other blocks on top of it and if the block is not picked up.
I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.
I can only unstack a block from on top of another block if the block I am unstacking is clear.
Once I pick up or unstack a block, I am holding the block.
I can only put down a block that I am holding.
I can only stack a block on top of another block if I am holding the block being stacked.
I can only stack a block on top of another block if the block onto which I am stacking the block is clear.
Once I put down or stack a block, my hand becomes empty.

Here is the format of the actions:
pick up the [block_name] block
unstack the [block_name] block from on top of the [another_block_name] block
put down the [block_name] block
stack the [block_name] block on top of the [another_block_name] block

State Representation:
After each reasoning step, you must explicitly write the full state of the world.

The state is written using the following predicates:
on(X, Y): block X is on block Y or on the table
clear(X): block X has no block on top of it and is not being held
holding(X): you are holding block X
handempty: you are not holding any block

The state must list all facts that are true. Any fact not listed is assumed to be false.

[Problem]
Here is the initial state of the blocks: {question}
Here is the goal state of the blocks: {answer}.

Show your work using the following format:

<think>
Explain your reasoning step by step about how the blocks should be rearranged to reach the goal state.

After each reasoning step, explicitly write the complete current state of the world using a <state> block.
You may include as many <state> </state> blocks as needed within this <think> block.

Each <state> block must list **all and only** the facts that are true at that moment.
Any fact not listed is assumed to be false.

<state>
on(...)
clear(...)
holding(...) or handempty
</state>

(Repeat reasoning and <state> blocks as needed.)
</think>

After completing your reasoning, provide the final sequence of actions in <answer> </answer> tags, for example:

<answer>
unstack the cyan block from on top of the emerald block
put down the cyan block
</answer>

Do not output anything outside the specified tags.
"""

            prompt = [{"role": "user", "content": prompt.format(question=question, answer=answer)}]
            reward_model = {"style": "rule", "ground_truth": {"question": question, "answer": answer, "solution": solution}}

            data = {
                "data_source": 'blocksworld-state-inline',
                "prompt": prompt,
                "ability": "math",
                "reward_model": reward_model,
                "extra_info": {"split": split, "index": idx},
                "uuid": str(uuid.uuid4()),
            }
            return data

        return process_fn


    if args.is_eval == False:
        test_dataset = test_dataset.select(range(150))
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Filter test_dataset to only include specific indices, such as 5, 9, and 12
    # selected_indices = [5, 9, 12]
    # incorrectIndex = [4, 6, 9, 10, 13, 16, 17, 18, 19, 22, 27, 32, 37, 39, 40, 44, 49, 53, 56, 57, 61, 63, 66, 67, 72, 74, 76, 77, 79, 82, 83, 84, 85, 89, 91, 93, 94, 95, 96, 98, 100, 101, 102, 103, 109, 110, 111, 112, 115, 116, 118, 119, 120, 126, 127, 128, 129, 130, 131, 134, 135, 138, 139, 140, 141, 145, 147, 153, 154, 156, 158, 160, 161, 164, 166, 168, 175, 176, 178, 179, 180, 181, 186, 190, 191, 192, 193, 197, 200, 203, 204, 205, 206, 208, 209, 214, 215, 216, 218, 222, 226, 227, 228, 229, 230, 231, 232, 233, 236, 237, 239, 241, 244, 245, 246, 248, 252, 265, 268, 270, 272, 275, 278, 283, 295, 296, 297, 302, 303, 312, 316, 317, 319, 324, 325, 327, 331, 335, 336, 341, 345, 347, 348, 352, 353, 355, 357, 358, 362, 363, 364, 365, 366, 367, 370, 373, 380, 383, 385, 389, 390, 396, 399, 402, 403, 407, 408, 409, 411, 412, 415, 419, 426, 428, 429, 430, 433, 439, 440, 441, 444, 453, 457, 461, 462, 464, 466, 467, 470, 475, 477, 480, 482, 483, 486, 501, 502, 503, 507, 509, 510, 513, 514, 518, 525, 529, 530, 535, 536, 537, 539, 540, 544, 545, 548, 549, 552, 553, 555, 556, 558, 559, 562, 565, 566, 567, 568, 576, 580, 581, 582, 585, 589, 590, 591, 593, 594, 598, 599, 601, 603, 604, 606, 607, 608, 609, 610, 614, 617, 620, 625, 628, 629, 631, 632, 641, 643, 644, 645, 646, 647, 648, 649, 650, 658, 659, 660, 668, 670, 671, 674, 676, 678, 679, 680, 688, 689, 691, 693, 696, 697, 699, 706, 707, 711, 712, 714, 715, 716, 718, 719, 720, 730, 731, 732, 733, 734, 736, 737, 738, 739, 744, 751, 755, 759, 761, 763, 766, 770, 771, 772, 774, 778, 787, 789, 791, 792, 796, 798, 802, 804, 808, 810, 811, 812, 817, 818, 819, 821, 822, 824, 826, 828, 829, 830, 831, 832, 833, 834, 837, 839, 841, 843, 851, 852, 854, 860, 863, 865, 866, 869, 871, 873, 874, 875, 878, 879, 886, 888, 890, 892, 895, 896, 897, 899, 904, 908, 910, 919, 920, 922, 923, 926, 927, 929, 930, 931, 934, 935, 936, 943, 944, 945, 948, 952, 954, 962, 965, 966, 968, 970, 971, 973, 975, 976, 980, 982, 983, 986, 988, 989, 992]
    # correctIndex = [0, 1, 2, 3, 5, 7, 8, 11, 12, 14, 15, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31, 33, 34, 35, 36, 38, 41, 42, 43, 45, 46, 47, 48, 50, 51, 52, 54, 55, 58, 59, 60, 62, 64, 65, 68, 69, 70, 71, 73, 75, 78, 80, 81, 86, 87, 88, 90, 92, 97, 99, 104, 105, 106, 107, 108, 113, 114, 117, 121, 122, 123, 124, 125, 132, 133, 136, 137, 142, 143, 144, 146, 148, 149, 150, 151, 152, 155, 157, 159, 162, 163, 165, 167, 169, 170, 171, 172, 173, 174, 177, 182, 183, 184, 185, 187, 188, 189, 194, 195, 196, 198, 199, 201, 202, 207, 210, 211, 212, 213, 217, 219, 220, 221, 223, 224, 225, 234, 235, 238, 240, 242, 243, 247, 249, 250, 251, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 266, 267, 269, 271, 273, 274, 276, 277, 279, 280, 281, 282, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 298, 299, 300, 301, 304, 305, 306, 307, 308, 309, 310, 311, 313, 314, 315, 318, 320, 321, 322, 323, 326, 328, 329, 330, 332, 333, 334, 337, 338, 339, 340, 342, 343, 344, 346, 349, 350, 351, 354, 356, 359, 360, 361, 368, 369, 371, 372, 374, 375, 376, 377, 378, 379, 381, 382, 384, 386, 387, 388, 391, 392, 393, 394, 395, 397, 398, 400, 401, 404, 405, 406, 410, 413, 414, 416, 417, 418, 420, 421, 422, 423, 424, 425, 427, 431, 432, 434, 435, 436, 437, 438, 442, 443, 445, 446, 447, 448, 449, 450, 451, 452, 454, 455, 456, 458, 459, 460, 463, 465, 468, 469, 471, 472, 473, 474, 476, 478, 479, 481, 484, 485, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 504, 505, 506, 508, 511, 512, 515, 516, 517, 519, 520, 521, 522, 523, 524, 526, 527, 528, 531, 532, 533, 534, 538, 541, 542, 543, 546, 547, 550, 551, 554, 557, 560, 561, 563, 564, 569, 570, 571, 572, 573, 574, 575, 577, 578, 579, 583, 584, 586, 587, 588, 592, 595, 596, 597, 600, 602, 605, 611, 612, 613, 615, 616, 618, 619, 621, 622, 623, 624, 626, 627, 630, 633, 634, 635, 636, 637, 638, 639, 640, 642, 651, 652, 653, 654, 655, 656, 657, 661, 662, 663, 664, 665, 666, 667, 669, 672, 673, 675, 677, 681, 682, 683, 684, 685, 686, 687, 690, 692, 694, 695, 698, 700, 701, 702, 703, 704, 705, 708, 709, 710, 713, 717, 721, 722, 723, 724, 725, 726, 727, 728, 729, 735, 740, 741, 742, 743, 745, 746, 747, 748, 749, 750, 752, 753, 754, 756, 757, 758, 760, 762, 764, 765, 767, 768, 769, 773, 775, 776, 777, 779, 780, 781, 782, 783, 784, 785, 786, 788, 790, 793, 794, 795, 797, 799, 800, 801, 803, 805, 806, 807, 809, 813, 814, 815, 816, 820, 823, 825, 827, 835, 836, 838, 840, 842, 844, 845, 846, 847, 848, 849, 850, 853, 855, 856, 857, 858, 859, 861, 862, 864, 867, 868, 870, 872, 876, 877, 880, 881, 882, 883, 884, 885, 887, 889, 891, 893, 894, 898, 900, 901, 902, 903, 905, 906, 907, 909, 911, 912, 913, 914, 915, 916, 917, 918, 921, 924, 925, 928, 932, 933, 937, 938, 939, 940, 941, 942, 946, 947, 949, 950, 951, 953, 955, 956, 957, 958, 959, 960, 961, 963, 964, 967, 969, 972, 974, 977, 978, 979, 981, 984, 985, 987, 990, 991, 993, 994, 995, 996, 997, 998, 999]
    # incorrectIndex = incorrectIndex[:250]
    # correctIndex = correctIndex[:50]
    # selected_indices = incorrectIndex + correctIndex
    # test_dataset = test_dataset.select(selected_indices)
    # breakpoint()


    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if args.is_eval == False:
        train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
        test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    else:
        test_dataset.to_parquet(os.path.join(local_dir, "eval.parquet"))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
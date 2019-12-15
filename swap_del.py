import argparse
import random


def swap(word):
    w = list(word)
    i = random.randint(1, len(w) - 3)
    w[i], w[i+1] = w[i+1], w[i]
    return ''.join(w)

def delete(word):
    w = list(word)
    i = random.randint(1, len(w) - 2)
    w.remove(w[i])
    return ''.join(w)

def process_line(line, mode):
	words = line.strip().split()
	longer_words = [w for w in words if len(w) > 3]
	try:
		random_word = random.choice(longer_words)
	except IndexError as e:
		print('Skipping line because it has no long words: \n {}'.format(line))
		return

	if mode == 'swap':
		new_word = swap(random_word)
	else:
		new_word = delete(random_word)
	words[words.index(random_word)] = new_word

	return ' '.join(words)


def main(input_file, output_file):
	lines = [line.strip() for line in open(input_file,'r')]
	out = open(output_file, 'w')

	i = 0
	for line in lines:
		if (i % 2 == 0):
			new_line = process_line(line, 'swap')
		else:
			new_line = process_line(line, 'delete')
		if i == len(lines) - 1:
			out.write("{}".format(new_line))
		else:
			out.write("{}\n".format(new_line))
		i += 1


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Perform swap and delete permutations on sentences.')
	parser.add_argument('input_file', type=str,
                    help='The file to be processed.')
	parser.add_argument('output_file', type=str,
                    help='The name of the file in which to store the newly created lines.')

	args = parser.parse_args()
	main(args.input_file, args.output_file)

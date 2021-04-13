import cmudict
import panphon.distance

NO_OF_PRECEDING_LINES = 3
IPA_VOWELS = {'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
              'ɪ', 'ʏ', 'ɪ̈', 'ʊ̈', 'ʊ',
              'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
              'ə',
              'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
              'æ', 'ɐ',
              'a', 'ɶ', 'ɑ', 'ɒ'}
ARPA_VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY',
               'EH', 'ER', 'EY',
               'IH', 'IX', 'IY',
               'OW', 'OY',
               'UH', 'UW', 'UX'}
SIMILAR_SOUNDS = [['B'], ['CH'], ['D'], ['DH'], ['DX'], ['EL'], ['EM'], ['EN'], ['F'], ['G'], ['HH'],
                  ['JH'], ['K'], ['L'], ['M', 'N', 'NG'], ['P'], ['Q'], ['R'], ['S'], ['SH'], ['T', 'TH'],
                  ['V'], ['W'], ['WH'], ['Y'], ['Z'], ['ZH'],
                  ['AA', 'AO'], ['AA', 'AH'], ['AE'], ['AW'], ['AX'], ['AXR'], ['AY'],
                  ['EH'], ['ER'], ['EY'],
                  ['IH'], ['IX'], ['IY'],
                  ['OW', 'AO'], ['OY'],
                  ['UH'], ['UW'], ['UX']]
dst = panphon.distance.Distance()
dict = cmudict.dict()
# Examples of input and output.
poem1 = ['Roses are red', 'you are too', "please don't be mad", 'be a fool.']
poem2 = ["Twinkle, twinkle, little star,", "How I wonder what you are.", "Up above the world so high,", "Like a diamond in the sky.", "When the blazing sun is gone,", "When he nothing shines upon,", "Then you show your little light,", "Twinkle, twinkle, all the night."]
lyrics1 = ["We were both young when I first saw you.", "I close my eyes and the flashback starts:", "I'm standing there", "On a balcony in summer air.", "See the lights, see the party, the ball gowns,", "See you make your way through the crowd,", "And say, Hello.", "Little did I know...",
           "That you were Romeo, you were throwing pebbles", "And my daddy said, Stay away from Juliet.", "And I was crying on the staircase", "Begging you, Please don't go.", "And I said,", "Romeo, take me somewhere we can be alone.", "I'll be waiting. All there's left to do is run.",
           "You'll be the prince and I'll be the princess.", "It's a love story. Baby, just say 'Yes'."]
# scheme: a, b, c, c, d, e, f, f, g, h, i, j, k, l, m, n, n
lyrics2 = ["I'm at a party I don't wanna be at", "And I don't ever wear a suit and tie, yeah", "Wonderin' if I could sneak out the back", "Nobody's even lookin' me in my eyes", "Can you take my hand?", "Finish my drink, say, Shall we dance?", "You know I love ya, did I ever tell ya?",
           "You make it better like that", "Don't think I fit in at this party", "Everyone's got so much to say", "I always feel like I'm nobody", "Who wants to fit in anyway?", "Cause I don't care when I'm with my baby, yeah", "All the bad things disappear"]
# a, a, b, a, c, d, a, c, e, f, e, f, g, g
lyrics2_syllables_correct = [['wi', 'wər', 'boʊθ', 'jəŋ', 'wɪn', 'aɪ', 'fərst', 'sɔ', 'ju'],
                             ['aɪ', 'kloʊz', 'maɪ', 'aɪz', 'ənd', 'ðə', 'ˈflæʃ', 'ˌbæk', 'stɑrts'],
                             ['əm', 'ˈstæn', 'dɪŋ', 'ðɛr'],
                             ['ɔn', 'ə', 'ˈbæl', 'kə', 'ni', 'ɪn', 'ˈsə', 'mər', 'ɛr'],
                             ['si', 'ðə', 'laɪts', 'si', 'ðə', 'ˈpɑr', 'ti', 'ðə', 'bɔl', 'gaʊnz'],
                             ['si', 'ju', 'meɪk', 'jʊr', 'weɪ', 'θru', 'ðə', 'kraʊd'],
                             ['ənd', 'seɪ', 'hɛˈ', 'loʊ'],
                             ['ˈlɪ', 'təl', 'dɪd', 'aɪ', 'noʊ'],
                             ['ðət', 'ju', 'wər', 'ˈroʊ', 'mi', 'ˌoʊ', 'ju', 'wər', 'θro', 'ʊɪŋ', 'ˈpɛ', 'bəlz'],
                             ['ənd', 'maɪ', 'ˈdæ', 'di', 'sɛd', 'steɪ', 'əˈ', 'weɪ', 'frəm', 'ˈʤu', 'li', 'ˌɛt'],
                             ['ənd', 'aɪ', 'wɑz', 'kraɪ', 'ɪŋ', 'ɔn', 'ðə', 'ˈstɛr', 'ˌkeɪs'],
                             ['ˈbɛ', 'gɪŋ', 'ju', 'pliz', 'doʊnt', 'goʊ'],
                             ['ənd', 'aɪ', 'sɛd'],
                             ['ˈroʊ', 'mi', 'ˌoʊ', 'teɪk', 'mi', 'ˈsəm', 'ˌwɛr', 'wi', 'kən', 'bi', 'əˈ', 'loʊn'],
                             ['aɪl', 'bi', 'ˈweɪ', 'tɪŋ', 'ɔl', 'ðɛrz', 'lɛft', 'tɪ', 'du', 'ɪz', 'rən'],
                             ['jul', 'bi', 'ðə', 'prɪns', 'ənd', 'aɪl', 'bi', 'ðə', 'ˈprɪn', 'sɛs'],
                             ['ɪts', 'ə', 'ləv', 'ˈstɔ', 'ri', 'ˈbeɪ', 'bi', 'ʤɪst', 'seɪ', 'jɛs']]


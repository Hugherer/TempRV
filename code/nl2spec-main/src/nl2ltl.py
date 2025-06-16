'''
this is changed by hugh !!!
which is used for Natural Language trans to LTL 
'''

import parser
import backend


def nl2ltl_main():
    args = parser.parse_args()
    res = backend.call(args)
    print("Final formalization with confidence score:")
    print(res[0])
    print("Sub-translations with confidence scores:")
    print(res[1][:-1])  # Remove the information whether a subtranslation is locked
    return res[0][0]

'''
if __name__ == "__main__":
    main()
'''

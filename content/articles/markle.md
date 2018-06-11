Title: Markle Tree in Python
Slug: markle_tree
Date: 2018-06-10 12:00
Category: Blockchain
Tags: Cryptography
Author: Tomoaki Fujii
Status: published




Blockchain has been attracting a lot of attention especially in a recent few years. This hype is backed by the belief
that information has been communicated in more secure way. As one of the way, we can consider digital signature system.
This technology has made sure that signature awesome.

# Background
## Cryptography
[Cryptography](https://en.wikipedia.org/wiki/Cryptography) is referred to the study of secure communication. One of the
most widely used methods is [RSA](https://sites.math.washington.edu/~morrow/336_09/papers/Yevgeny.pdf),
which is based on characteristic of prime numbers. This method take advantage of difference computational difficulties
between multiplication and prime decomposition. 
* [Nice Video on Youtube](https://www.youtube.com/watch?v=vgTtHV04xRI)
* My python implementation is here.

There is another method is [EDCSA](https://blog.cloudflare.com/a-relatively-easy-to-understand-primer-on-elliptic-curve-cryptography/).
This is method is still controversial, but known to be more secure and fast.

Both RSA amd EDCSA utilizes the idea of using a pair of private and public key for secure communication. Theses methods
are executed as the following:

#### Set up
1. Generate a pair of public and private keys
2. Distribute private key to someone to communicate with while keep private key in your side

#### Communication
1. Encrypt a message using public key and send the encrypted message to you
2. Decrypt the received encrypted message using the private key

Note that by not sharing the private key, you can make sure the secure communication. What you have to remeber is
* Public key is used for encryption and shared 
* Private key is used for decryption and not shared


When sending method, we access to public key which is distributed in public
as in name and encrypt your message. Then, the reciever decrypt the encrypted message using private key. 

## Digital Signature
Digital signature is one of the methods to make sure who send the message you recieve. The process is the followings:
* Sign the message and send
* Recieve and decrypto with public key

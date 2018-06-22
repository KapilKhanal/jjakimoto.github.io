Title: Toward Understanding Blockchain
Slug: blockchain
Date: 2018-06-22 12:00
Category: Blockchain
Tags: Blockchain
Author: Tomoaki Fujii
Status: published

Blockchain is one of the hottest technologies as well as AI/Machine Learning. Indeed,
a lot of startup are working in this fields. Besides that compared to another hot topic, Deep Learning, Blockchain is 
still in early stage and there is space you can get in [[1]](https://medium.freecodecamp.org/the-authoritative-guide-to-blockchain-development-855ab65b58bc).
I know that a lot of folks investing in cryptocurrencies don't give a shit about tech behind it like Blockchain, but if
you are savvy tech guy, it would be good investment to learn this topic. 

![blockchain_ml]({filename}/images/blockchain/blockchain_ml.png)

Blockchain ranges across various topics such as cryptography and distributed system. In this blog, I briefly go through this
tipic while suggesting various blogs and papers that I have used for learning this topic. Especially, I highly recommend
you to check a blog post [[1]] written by Haseeb Qureshi. Let's dig into Blockchain!


# 1. Cryptography
Cryptography [[2]](https://en.wikipedia.org/wiki/Cryptography) is referred to the study of secure
communication. The goal is how we send messages to each other without revealing them to someone else.
One of the most widely used methods is RSA [[3]](https://sites.math.washington.edu/~morrow/336_09/papers/Yevgeny.pdf),
which utilizes properties of prime numbers. This method take advantage of difference in computational difficulties
between multiplication and prime decomposition. 
* Nice explanation on Youtube [[4]](https://www.youtube.com/watch?v=vgTtHV04xRI)
* Python Implementation [[5]](https://gist.github.com/JonCooperWorks/5314103)

There is another method called EDCSA [[6]](https://blog.cloudflare.com/a-relatively-easy-to-understand-primer-on-elliptic-curve-cryptography/).
This is method is still controversial, but known to be more secure and fast.

## Encryption and Decryption
To achieve secure communications, we need to encrypt data before sending and decrypt received data.
In the early age, people share secrete information about how to encrypt or decrypt in some way. Then,
they communicate to each other. This method has a risk that the secret information is stolen while sharing.

To reconcile this problem, the idea of asymmetric key is suggested. Both RSA amd EDCSA are based on this
idea. Asymmetric key consists of public key and private key. Let's look into how this concept works.

#### 1. Set up
1. Generate a pair of public and private keys
2. Distribute the public key to someone to communicate with while keeping private key in only your side

#### 2. Communication
1. Someone encrypts a message using the public key and send the encrypted message to you
2. Decrypt the received encrypted message using the private key

Note that by not sharing the private key, you can make sure the secure communication. What you have
to remember is
* Public key is used for encryption and shared 
* Private key is used for decryption and not shared

## Digital Signature
Digital signature is one of the methods to identify who sends the message. This method utilizes the idea
of asymmetric key explained above. The process is the following ways.

#### 1. Sign the message
* Encrypt message using private keys: $Signature = Sign(Message, PrivateKey)$

#### 2. Verify the signature
* Decrypt the signature: $DecryptedSignature = Func1(Signature)$
* Make correspondence output from the private key and the message: $Output = Func(Message, PublicKey)$
* See if $DecryptedSignature$ and $Output$ are matched up

Note that unlike secure communications, we open the message itself to public. Receiver only has to
know what output would be, which is produced by public key and the message.

#### Merkle Tree
Merkle Tree is one of efficient digital signature algorithms, which enables Blockchain scalable.
The signature proof is based on executed with binary tree. You can find a simple explanation of this
algorithm at this blog [[7]](https://hackernoon.com/merkle-tree-introduction-4c44250e2da7). I also recommend
you to implement Markle Tree by yourself to understand how it works [[8]](https://github.com/evankozliner/merkle-tree/blob/master/MerkleTree.py).

For more detail explanation about digital signature, please refer to this paper [[9]](https://www.emsec.rub.de/media/crypto/attachments/files/2011/04/becker_1.pdf).




# 2. Distributed System
Distributed system often appears in the context of high performance computing and scalable database. 
As an introduction, you can refer to the paper [[10]](https://link.springer.com/content/pdf/10.1007%2Fs00607-016-0508-7.pdf).
In this section, we go though the basic concept and how to achieve such systems.


## CAP Theorem
When discussing what distributed systems have to achieve, the following three
properties are usually mentioned:
* Consistency
* Availability
* Partition Tolerance

Gilbert and Lynch [[11]](https://www.glassbeam.com/sites/all/themes/glassbeam/images/blog/10.1.1.67.6951.pdf)
shows that it is impossible to achieve these three properties at the
same time. This theory is called `CAP Theorem` after initials of three properties. Let's look
into more precise definitions.

#### Consistency
Gilbert and Lynch describe `Consistency` as 
> any read operation that begins after a write operation completes must return that value, or the result of a later write operation

Let's say there are two nodes A and B.
We write file F to node A followed by the reading F query to B after 1.0 seconds.
Fetched information from node B is matched up with stored information at A
if B is updated within one seconds, otherwise not. Thus, depending on frequency of updates, the response
may not be consistent.
 
Keeping consistency is important to build a reliable system.

#### Availability
Gilbert and Lynch describe `Availability` as 
> every request received by a non-failing node in the system must result in a response

It simply means to keep responding to clients without interruption.
For example, if we have two node A and B. Even if node A has collapsed, we are able to response with
node B. Hence, this system has availability up to collapse of one of the nodes. 

#### Partition Tolerance
Gilbert and Lynch describe `Partition Tolerance` as 
> the network will be allowed to lose arbitrarily many messages sent from one node to another

Some parts of your system may corrupt temporary and break connections among
nodes. Then, some sent messages are unable to reach their destinations.
`Partition Tolerance` means allowing your system to keep running even in such situations.
 
 
At most, two out of these three properties are able to achieve at the same time.
You can find more intuitive proof at a nice blog post [[12]](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/).

Practically speaking, you have to choose `Partition Tolerance` and either of `Availability` or `Consistency`.
As the size of your system increases, some failure is more likely to happens.
Thus, It is not practical consider a system without `Partition Tolerance`. More precise
explanation is found in this blog post [[13]](https://codahale.com/you-cant-sacrifice-partition-tolerance/).


## Networking
To establish a distributed system, nodes have to communicate to each other.
Blockchain adopts pear-to-pear networking [[14]](https://en.wikipedia.org/wiki/Peer-to-peer).

#### Pear-to-Pear Networking
Pear-to-pear networking has the following properties [[15]](https://www.digitalcitizen.life/what-is-p2p-peer-to-peer):

* Act as both client and server
* Share resources (e.g., processing power, disk storage and network bandwidth)
* Fast file sharing
* Difficult to take down
* Scalable

These properties make it possible to build robust distributed system.
Napster, music file sharing service, is one of the first applications of pear-to-pear network system.

#### Internet Protocol (IP)
There are two types of IP traffic: TCP (Transmission Control Protocol) and UCP
(User Datagram Protocol). Blockchain uses TCP. Let's see the difference between them.

##### TCP
* Connection-oriented protocol
* Rearranges data packets in the order specified
* Slower than UDP
* Grantees that data transfers correctly
* HTTP, HTTPs, FTP, etc.
* Suitable for applications that require high reliability
* Example: email

##### UDP
* Connectionless protocol
* No inherent order
* Faster because of no error recovery attemption
* No grantee of transfers
* DNS, DHCP, TFTP, etc.
* Suitable for applications that need fast and efficient transmission
* Example: game

For more detail, please refer to this blog post [[16]](https://www.diffen.com/difference/TCP_vs_UDP).


#### Gossip Protocol
A gossip protocol [[17]](https://en.wikipedia.org/wiki/Gossip_protocol) is a 
procedure or process of computer-computer communication.
It takes the following steps:

1. Pick up a pear at random from nodes
2. Communicate with the chosen pear









# 3. Blockchain
Now we finally come to Blockchain!. Blockchain allows you to authenticate transactions in decentralized fashion.
Blockchain mainly consists of
* Verification: Check if a transaction is valid
* Consensus: Match up transaction history through voting system

Verification includes simply verifying who send this transaction or if transactions are over the holding amounts, etc.
In the consensus process, we need to communicate with others and try to agree on what transactions would be included.
Consensus works in the global scale while verification works locally. Let's look into more detail. 

## Verification
Digital signature can be used to verify who sends messages you have received. In the case
of Blockchain, we verify who sends a transaction.
To figure how it works, let's start from a simple case: a transaction between only two people.

#### Two people case
When you lend or borrow money,
you may record how much and when somewhere like in a notebook. If this transaction happens among reliable friends, this
setting is good enough. This, however, is not necessary the case for general cases. Usually, you need someone else
to validate the record. For traditional currencies, the third person would be some financial institutes like bank. They
monitor transactions if they are valid and record to their database.

Blockchain uses digital signature to validate transactions instead of introducing a third person. Let's say a transaction
happens between Adnan and Barby. The transaction looks like `Adnan is going to give an apple to Barby`.
To set up digital signature,
they first need to generate their own pairs of private and public keys, and hand their public keys to
each other. To make sure the transaction, they take the following steps:
1. Adnan writes a transaction in plaintext saying `From: Adnan, To: Barby, What: 1 apple`
2. Adnan encrypts the plaintext with his private key.
3. Adnan prepends a plaint text "singed by Adnan" to the ciphertext and send it to Barby
4. Barby receives the text and finds "signed by Adnan"
5. Barby decrypts the ciphertext with the public key of Adnan to verify if the transaction comes from him

Note that the processes 3 and 4 are necessary when more than two parties are involved. Otherwise, Barby is
unable to decide whose public key she has to use.


#### Ledger
Transactions like above are recorded something called ledger. Ledger consists of multiple transactions.
Each transaction contains a ciphertext and a plaintext about whose sign. By using corresponding public keys,
you can decrypts ciphertext and calculate the balance of the ledger. 


#### Multi-party transfers and verification
Let's consider the case where three people are involved: Adnan, Barby, and Carl. They exchange some valuable coins.
Assume that they have the following transactions:

* Transaction 1. `From: Barby, ciphertext 1 => To: Adnan, What: 3 coins`
* Transaction 2. `From: Adnan, ciphertext 2 => To: Carl, What: 3 coins`

You can translate this situation to that Barby pays 3 coins to Carl. Thus, Adnan can add another
transaction to the ledger:
* Transaction 3. `From Adnan, ciper_3 => To: Carl, What: Hashed value of chipertext 1`

Here are what Carl is going to do:
1. Fetch Baby's and Adnan's public keys and verifies Transaction 1 and 2.
2. Verify that Adnan transfers a valid transaction:
    * The transaction is addressed to Adnan
    * Adnan has not previously transfered the same transaction to anyone else

After all checks pass, Carl asks Baby to pay.
Then, here are what Baby is going to do:
1. Fetch Adnan's public key and verify Transaction 3
2. Verify that the transfer is referencing one of her own transaction with Adnan


By utilizing digital signatujre Blockchain allows you to have secure transaction. There, however, remains
some flaws in this system. When validating transaction, the reference, i.e., the ledger is not necessary updated.
If someone makes multiple transactions almost at the same time, they are able to use the same transaction
more than once. Let's discuss thie issue at the next section.


## Consensus
Sharing the same ledger is important when verifying transactions. Inconsistency allows malicious users
to fraud transactions. The typical fraud transaction is double-spending, spending the same transaction twice.

#### Double-spending and distributed consensus
In the previous chapter, Adnan transfers Barby's transaction to Carl. If Adnan moves quickly before
someone has not updated their ledger, he can transfer the same transaction to them again. This is
called a double-spending attack. To combat this situation, we need to consider distributed consensus
system [[2]](https://en.wikipedia.org/wiki/Consensus_(computer_science)).


#### P2P Network
We consider consensus through voting system. Pear-to-pear (P2P) network comes in to establish
a such system that deals with two issuse: Geography and Timezone problems.

Thus, P2P network 
* Communicates through online (Geography)
* Updates ledger automatically (Timezone)
* Uses Gossip Protocol to update the ledger with voting system


There still remains some problems with out P2P design:
* Ensuring that every participant is always up to date imposes high coordination costs and affects availability:
if a single peer is unreachable the system cannot commit new transactions.(Availability)
* In practice, we do not know the global status of the P2P network: Some of nodes may have left the network.
(Partition Tolerance)
* The system is open to a Sybil attack [[sybil]](https://en.wikipedia.org/wiki/Sybil_attack). (Consensus)

According to the CAP Theorem, all of the problems are unable to solve at the same time.
Rather than taking either of CP or AP, we operate the P2P network under the weaker consistency[[weak]](https://www.igvita.com/2010/06/24/weak-consistency-and-cap-implications/),
which switch foreword and back between CP and AP.

This requires the network to deal with the followings:
* We must accept that some ledgers will be out of sync (at least temporarily).
* The system must eventually converge on a global ordering (linearizability) of all transactions.
* The system must resolve ledger conflicts in a predictable manner.
* The system must enforce global invariants, e.g., no double-spends.
* The system should be secure against Sybil and similar attacks

Roughly speaking, out goal is building the network preventing fraud while keeping consistency
to some extent. Then, proof-of-work comes in to help you build such system.

#### Proof-of-work
Blockchain introduces proof-of-work to avoid Sybil attacks.
The goals of proof-of-work are
* "expensive" for the sender.
* "cheap" to verify by everyone else.

These goals make Sybil attacks expensive and get rid of their economical benefits while keeping running
the secure system with reasonable cost. Let's see how these goals are achieved.

* Process of sender: To search an input of a hash function to produce an output satisfying a certain rule
> Let's say you can set a rule like first four characters have to be 0, e.g., "0000A9E5F3". Guessing the input of a hash function from given the output is difficult. You need to resort to brute force search.
Thus, it takes for a while to find an ideal input value.

* Process of verification: To hash transaction and see if hashed value is matched up with the output
> This process is computationally cheap.



In Blockchain, proof-of-work works in the following way,
1. Collect new transactions
2. Keep trying different $Nonce$ [[nonce]](https://en.wikipedia.org/wiki/Cryptographic_nonce) until $hash_value$ of the below equation satisfies given rule
$$hash_func(Transactions + Nonce + Signature + HashValue_PreviousBlock) => HashValue$$

After proof-of-work has finished, you add the block to a chain called Blockchain.
Blockchain is constructed from blocks verified by proof-of-work. Including previous hash values avoids someone to change
the transaction records in the middle of the chain. 

You can get more intuitive comprehension about how proof-of-work in Blockchain from [this demonstration](https://youtu.be/_160oMzblY8).


#### Blockchain
We briefly looked how proof-of-work prevent Sybill attack and build the chain of transaction blocks, Blockchain.
Now, we are going to see the detail as to how Blcockchain are  built while preventing frauds. 


###### Issues in validation
Since our P2P network has to be scalable and dynamic, we have to deal with the following situations:
* We do not know how many people to contact to get their approval
* We do not know whom to contact to get their approval
* We do not know whom we are calling

Lacking identity and global knowledge of all participants in the network does not allow you to grantee
that any transaction is valid. We, however, are probabilistically able to grantee the validation.

###### N-confirmation transaction
Contact N participants and get them to verify transactions. 
* The larger N, the higher likely transaction is valid
> If malicious users are less than half of participants, we can grantee the validity of confirmation
asymptotically.
* The larger costs you more proof-of-work
> Increasing costs deprive frauds of economical benefits. As a result, we exepect more participants
are less likely to work on frauds.

The optimal value of N would be determined by the trade-off between the secure and cost of confirmation. 



###### Adding blocks and transaction fee incentives
We do not give any reasons for someone to join and work on proof-of-work. Then, we give incentive to someone succeeded in
proof-of-work. This incentive, however, has to be small enough. Otherwise, it holds participants to have transactions. 
Miners bascially collect a lot of transactions and then get incentive from there. 
Let's see how the process works.

1. Adnan and Barby generate new transactions and announce therm to the network.
2. Carl is listening for new transactions with their incentives. Then he works on the following processes:
    * Collect unconfirmed transactions until total incentives is large enough (Incentives have to be higher than
    proof-of-work)
    * Verify all of the collected transactions and add a new transactions that transfers incentives to him
    * Work on proof-of-work and generate a validated block
    * Distribute the block to all other participants
3. Both Adnan and Barby are listening for new block announcements and look for their transaction in the list
    * Adnan and Barby verify integrity of the block
    * If the block is valid and their transactions are in the list, then the transactions are confirmed

###### Racing to claim the transaction fees
You may wonder what if more than one participants work on proof-of-work on the same transactions. Indeed, this situation
happens all the time and we have to consider the solution to integrate them. In Blockchain, the first one to finish proof-of-work
takes the all. Here is how it works:

1. Collect unconfirmed transactions and start proof-of-work
2. Generate a valid bock and broadcast it into the network
    * Other peers check the validity of the block
    * If the block is valid, it is added to the participants' ledgers and rebroadcast to other peers
    * Once the new block is added, abort their previous work
3. Repeat 1 and 2

By the nature of the race, the more computational power are more likely to be succeeded to finish proof-of-work. Although
participation in the race itself is for free, how much influence you give to building the chain is determined by how much
you put the cost to computational power. 

###### Resolving chain conflicts
If could happen that more than one participants find valid bocks almost at the same time and add blocks on top of the chain.
In that case, which blocks would be chosen as top-most block for the next proof-of-work?

In this situation, Blockchain takes the policy:
> The longest branch is always valid

Everytime you find a longer branch, you need to switch to that branch. Thus, `any blokcs are never 'final'!`. Prctically speaking,
we are unable to wait for infinite time to confirm the trasaction is valid. Mostly, we set certain number N to confirm transations.
The larger N gives you more security while taking a lot of time to confirm the transactions.

This policy also makes sense to avoid confirming fraud transactions. If less than half of miners are working on fraud. The
speed of devloping valid branch is faster than fraud branch probabilistically. Thsu, if N is large enough, we can make sure
that the branch contains the valid transaction. 

###### Properties of Blockchain
1. Individual transactions are secured by digital signature
2. Once created, transactions are broadcast into P2P network
3. One or more transactions are aggregated into a block
4. Peers listen for new block announcements and merge them into their ledgers

You can also check the nice explanation at [You Tube](https://youtu.be/bBC-nXj3Ng4)



# 4. Wrap Up
That's it! We walked through the basic of Blockchain. To deepen your comprehension, I highly recommend
to build your Blockchain, [Hasebi's video](https://youtu.be/3aJI1ABdjQk). Since this field changes really
fast, it's important to keep you updated with some media. Some platform are recommended in [Hasebi's blog](https://medium.freecodecamp.org/the-authoritative-guide-to-blockchain-development-855ab65b58bc).

# 5. References
1. [The authoritative guide to blockchain development](https://medium.freecodecamp.org/the-authoritative-guide-to-blockchain-development-855ab65b58bc)

2. [Cryptography (Wkikpedia)](https://en.wikipedia.org/wiki/Cryptography)
3. [The RSA Algorithm, E. Milanov](https://sites.math.washington.edu/~morrow/336_09/papers/Yevgeny.pdf)
4. [Gambling with Secrets: 8/8 (RSA Encryption) (YouTube)](https://www.youtube.com/watch?v=vgTtHV04xRI)
5. [RSA Python Implementation](https://gist.github.com/JonCooperWorks/5314103)
6. [A (Relatively Easy To Understand) Primer on Elliptic Curve Cryptography](https://blog.cloudflare.com/a-relatively-easy-to-understand-primer-on-elliptic-curve-cryptography/)
7. [Merkle Tree Introduction](https://hackernoon.com/merkle-tree-introduction-4c44250e2da7)
8. [Merkle Tree Python Implementation](https://github.com/evankozliner/merkle-tree/blob/master/MerkleTree.py)
9. [Merkle Signature Schemes, Merkle Trees and Their Cryptanalysis, G. Becker](https://www.emsec.rub.de/media/crypto/attachments/files/2011/04/becker_1.pdf)

10. [A brief introduction to distributed systems](https://link.springer.com/content/pdf/10.1007%2Fs00607-016-0508-7.pdf)
11. [Brewer’s Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web](https://www.glassbeam.com/sites/all/themes/glassbeam/images/blog/10.1.1.67.6951.pdf)
12. [An Illustrated Proof of the CAP Theorem](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/)
13. [You Can’t Sacrifice Partition Tolerance](https://codahale.com/you-cant-sacrifice-partition-tolerance/)
14. [Peer-to-peer (Wikipedia)](https://en.wikipedia.org/wiki/Peer-to-peer)
15. [Simple questions: What is P2P (peer-to-peer) and why is it useful?](https://www.digitalcitizen.life/what-is-p2p-peer-to-peer)
16. [TCP vs. UDP](https://www.diffen.com/difference/TCP_vs_UDP)
17. [Gossip protocol](https://en.wikipedia.org/wiki/Gossip_protocol)

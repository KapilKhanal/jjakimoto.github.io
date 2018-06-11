# Dsitributed System
Distributed system is one of the key concepts of blockchain, which apperears in
context of high performancde computing and scalable database. 
As an introduction, you can refer to [the paper](https://link.springer.com/content/pdf/10.1007%2Fs00607-016-0508-7.pdf).



## CAP Theorem
When discussing what distributed systemes have to achieve, the following three
properties are usually mentioned:
* Consistency
* Availability
* Partition Tolerance

In [the paper](https://www.glassbeam.com/sites/all/themes/glassbeam/images/blog/10.1.1.67.6951.pdf),
Gilbert and Lynch shows it is impossible to achieve these three properties at the
same time. This theory is called `CAP Theorem` by taking their initials. Let's look
into more precise definitions.

#### Consistency
Gilbert and Lynch describe `Consistency` as 
> any read operation that begins after a write operation completes must return that value, or the result of a later write operation

Let's say there are two nodes A and B.
We write file F to node A followed by the reading F query to B after 1.0 seconds.
Fetched information from node B is matched up with stored information at A
if B is updated within one seconds, otherwise not. Keeping consistency is one of
the important qualities to build a reliable system.

#### Availability
Gilbert and Lynch describe `Availability` as 
> every request received by a non-failing node in the system must result in a response

Simply, it means to keep responding to clients without interruption.

#### Partition Tolerance
Gilbert and Lynch describe `Partition Tolerance` as 
> the network will be allowed to lose arbitrarily many messages sent from one node to another

Some parts of your system may corrupt temporary and break connections among
nodes. Then, some messages sent in this situation are unable to reach the  destination.
`Partition Tolerance` means allowing your system to keep running even in such situations.
 
 
At most, two out of these three  properties are able to achieve at the same time.
You can find more intuitive proof at [the blog post](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/).

Practically speaking, you have to choose `Partition Tolerance` and either of 
`Availability` or `Consistency`.
As the size of your system increases, some failure is more likely to happens.
Thus, It is not practical consider a system without `Partition Tolerance`. More precise
explanation is found in [a blog post](https://codahale.com/you-cant-sacrifice-partition-tolerance/).


## Networking
To establish a distributed system, it requires for nodes to communicate to each other.
In blockchain, [pear-to-pear networking](https://en.wikipedia.org/wiki/Peer-to-peer)
is applied. Let's see them more pricisely.

#### Pear-to-Pear Networking
Pear-to-pear networking has [the following properties](https://www.digitalcitizen.life/what-is-p2p-peer-to-peer):

* Act as both client and server
* Share resources (e.g., processing power, disk storage and network bandwidth)
* Fast file sharing
* Difficult to take down
* Scalable

These properties makes it possible to build robust distributed system.
Napster, music file sharing service, is one of the first applications of pear-to-pear network system.

#### Internet Protocol (IP)
There are two types of IP traffic: TCP (Transmission Control Protocol) and UCP
(User Datagram Protocol). Blockchain uses TCP. Let's see the difference between them.

###### TCP
* Connection-oriented protocol
* Rearranges data packets in the order specified
* Slower than UDP
* Grantees that data transfers correctly
* HTTP, HTTPs, FTP, etc.
* Suitable for applications that require high reliability
* Example: email

###### UDP
* Connectionless protocol
* No inherent order
* Faster because of no error recovery attemption
* No grantee of transfers
* DNS, DHCP, TFTP, etc.
* Suitable for applications that need fast and efficient transmission
* Example: game

For more detail, you can refer to [a blog post](https://www.diffen.com/difference/TCP_vs_UDP).


#### Gossip Protocol
A [gossip protocol](https://en.wikipedia.org/wiki/Gossip_protocol) is a 
procedure or process of computer-computer communication.
It takes the following steps:

1. Pick up a pear at random from nodes
2. Communicate with the chosen pear


from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''


def get_twitter():
    
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)



def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.
    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.
    Here's a doctest to confirm your implementation is correct.
    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    with open('candidates.txt','r') as file:
        listOfCandidates = [line.rstrip('\n') for line in file]
    return listOfCandidates

def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)
    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup
    In this example, I test retrieving two users: twitterapi and twitter.
    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    ###TODO
    #comma_separated_string = ",".join(screen_names)
    #twitter = get_twitter()
    for screen_name in screen_names:
        request = robust_request(twitter,'users/lookup',{'screen_name':screen_names})
        users = [r for r in request]
        aList = []
        i = 0
        while i < len(users):
            aList.append(users[i])
            i = i + 1
        return aList
    
    
def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids
    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.
    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.
    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.
    In this test case, I return the first 5 accounts that I follow.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    #for i in screen_name:
    request = robust_request(twitter,'friends/ids', {'screen_name':screen_name, 'count':5000})
    friends = [r for r in request]
    #friend = friends.sort()
    listOffriends = [int(x) for x in friends]
    listOffriends.sort()

    return listOffriends
    

    
def add_all_friends(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.
    Store the result in each user's dict using a new key called 'friends'.
    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing
    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    for i in users:
        i['friends'] = get_friends(twitter, i['screen_name'])
    return None
        
def print_num_friends(users):
    
    for i in users:
        
        print("screen_name=%s, %d" %(i['screen_name'],len(i['friends'])))
    """Print the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    
def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter
    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    
    """
    c = Counter()
    for acandidate in users:
        c.update(acandidate['friends'])
    return c

def draw_network(graph, users, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).
    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.
    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    ###TODO
    
    nodeswiththepositions = dict()
    nodesnames = []
    nodesf = []
    for i in users:
        nodeswiththepositions[i['screen_name']] = i['screen_name']
    nx.draw_networkx(graph, pos=nx.circular_layout(graph, dim=2, scale=1), labels = nodeswiththepositions)
    plt.axis('off')
    plt.figure(1, figsize=(25, 25), dpi= 1048)
    plt.savefig(filename)
    
def create_graph(users, friend_counts):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)

        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.

    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    graph = nx.DiGraph()
    for u in users:
        graph.add_node(u['screen_name'])
        for friends_id in u['friends']:
            if friend_counts[friends_id] > 1:
                graph.add_node(friends_id)
                graph.add_edge(u['screen_name'],friends_id)
    return graph



def followed_by_hillary_and_donald(users, twitter):
    """
    Find and return the screen_name of the one Twitter user followed by both Hillary
    Clinton and Donald Trump. You will need to use the TwitterAPI to convert
    the Twitter ID to a screen_name. See:
    https://dev.twitter.com/rest/reference/get/users/lookup

    Params:
        users.....The list of user dicts
        twitter...The Twitter API object
    Returns:
        A string containing the single Twitter screen_name of the user
        that is followed by both Hillary Clinton and Donald Trump.
    """
    request = robust_request(twitter, 'users/show', {'user_id': 822215673812119553})
    req = [r for r in request]
    for r in req:
        return r['screen_name']

def friend_overlap(users):
    """
    Find and return the screen_name of the one Twitter user followed by both Hillary
    Clinton and Donald Trump. You will need to use the TwitterAPI to convert
    the Twitter ID to a screen_name. See:
    https://dev.twitter.com/rest/reference/get/users/lookup

    Params:
        users.....The list of user dicts
        twitter...The Twitter API object
    Returns:
        A string containing the single Twitter screen_name of the user
        that is followed by both Hillary Clinton and Donald Trump.
    """
    userList = []
    setFormationList = []
    overlaplist = []
    list_traversed = list()
    screen_names = dict()
    i = 0
    n = 0
    theFinalList = []
    
    #a = len(users)
    
    while i < len(users):
        userList.append(users[i]['friends'])
        i = i + 1
    b = len(userList)
    cd = len(users)
    zd = []
    qr = []
    smd = {}
    for q in range(0,len(userList)):
        for r in range(q+1,len(userList)):
            setFormationList.append(set(userList[q]).intersection(set(userList[r])))
    
    for q in range(0, len(userList)):
        for r in range(q+1, len(userList)):
            theFinalList.append((users[q]['screen_name'],users[r]['screen_name'],len(setFormationList[n])))
            n= n+1
    overlapp_list = sorted(theFinalList,key = lambda x: x[2],reverse = True)
    return overlapp_list


def main():
    twitter = get_twitter()    
    screen_names = read_screen_names("candidates.txt")
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
              (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))
    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')


if __name__ == '__main__':
    main()









class TrieNode:
    def __init__(self):
        self.children = {}  # 子节点映射
        self.isEndOfUrl = False  # 标记是否为URL的结尾

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, url):
        """插入一个URL到前缀树中"""
        node = self.root
        for char in url:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.isEndOfUrl = True  # 标记URL结束

    def search(self, url):
        """检查URL的前缀是否在前缀树中"""
        node = self.root
        for char in url:
            if char not in node.children:
                return False  # 没找到匹配的前缀
            node = node.children[char]
            if node.isEndOfUrl:
                return True  # 找到了匹配的前缀
        return node.isEndOfUrl  # 返回是否为完整的URL匹配
    

def remove_url_head(url):
    if url.startswith("http://"):
        return url[7:]
    if url.startswith("https://"):
        return url[8:]
    return url
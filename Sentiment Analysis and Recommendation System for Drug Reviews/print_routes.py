import app

def main():
    routes = []
    for r in app.app.url_map.iter_rules():
        routes.append((r.rule, sorted(list(r.methods))))
    out_lines = []
    for rule, methods in sorted(routes, key=lambda x: x[0]):
        out_lines.append(f"{rule} {'|'.join(methods)}")
    with open("routes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

if __name__ == "__main__":
    main()

/*+ HashJoin(t2 mk t1 lt ml k)
 HashJoin(mk t1 lt ml k)
 MergeJoin(mk t1 lt ml)
 HashJoin(t1 lt ml)
 NestLoop(lt ml)
 SeqScan(t2)
 SeqScan(mk)
 IndexScan(t1)
 SeqScan(lt)
 IndexScan(ml)
 SeqScan(k)
 Leading((t2 ((mk (t1 (lt ml))) k))) */
SELECT MIN(lt.link) AS link_type,
       MIN(t1.title) AS first_movie,
       MIN(t2.title) AS second_movie
FROM keyword AS k,
     link_type AS lt,
     movie_keyword AS mk,
     movie_link AS ml,
     title AS t1,
     title AS t2
WHERE k.keyword ='character-name-in-title'
  AND mk.keyword_id = k.id
  AND t1.id = mk.movie_id
  AND ml.movie_id = t1.id
  AND ml.linked_movie_id = t2.id
  AND lt.id = ml.link_type_id
  AND mk.movie_id = t1.id;

